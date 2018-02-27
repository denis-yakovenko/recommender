package recommender;

import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.*;

import java.io.IOException;
import java.util.*;

import static org.apache.spark.sql.functions.*;

public class ContextualPreAndPostFiltering {
    private static Dataset<Row> dataSet;
    private static StringIndexerModel itemIndexer;
    private static StringIndexerModel userIndexer;
    private static ALSModel model;
    private static SparkSession spark;
    private static Map<String, String> userContext = new HashMap<>();

    public static void main(String[] args) throws IOException {
        spark = SparkSession
                .builder()
                .enableHiveSupport()
                .master("local[*]")
                .appName("CARS")
                .getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");

        /* loading data set */
        //dataSet = Util.getDePaulMovieDataSet(spark);
        dataSet = Util.getDePaulMovieDataSetOriginal(spark);
        dataSet.cache();

        /* creating an example of the user context */
        userContext.put("Time", "Weekend");
        userContext.put("Location", "Home");
        userContext.put("Companion", "Family");

        /* showing dataSet total counts */
        Util.showDataSetStatistics(dataSet);

        /* indexing user, item and context columns by integer values */
        createIndexes();

        /* rescaling rating to range [0, 1] */
        dataSet = Util.getWithRescaledRating(dataSet);

        /* training the ALS model */
        trainModel();

        /* getting and showing predictions for the user with id 1001 using contextual pre-filtering */
        getPredictionsUsingPreFiltering("1001")
                .orderBy(desc("rating"))
                .show(false);

        /* getting and showing predictions for the user with id 1001 using contextual post-filtering */
        getPredictionsUsingPostFiltering("1001")
                .orderBy(desc("rating"))
                .show(false);

        spark.stop();
    }

    /**
     * getting predictions for the user with given id using contextual pre-filtering
     */
    private static Dataset<Row> getPredictionsUsingPreFiltering(String userId) {
        /* contextual pre-filtering of the source dataset using user context */
        /* In particular, in one possible use of this approach,
        context c essentially serves as a query for selecting (filtering) relevant ratings data. */
        Dataset<Row> dataSetFilteredWithContext = dataSet;
        for (Map.Entry<String, String> contextVar : userContext.entrySet()) {
            dataSetFilteredWithContext = dataSetFilteredWithContext.filter(col(contextVar.getKey()).equalTo(contextVar.getValue()));
        }

        /* getting list of the items for predict */
        dataSetFilteredWithContext.createOrReplaceTempView("dataSetFilteredWithContext");
        Integer userIdI = Arrays.asList(userIndexer.labels()).indexOf(userId);
        Dataset<Row> itemsToPredictRating = spark.sql("select distinct " + userIdI + " useridI, itemidI " +
                " from dataSetFilteredWithContext " +
                " where itemidI not in ( " +
                " select itemidI from dataSetFilteredWithContext where userIdI = " + userIdI + " " +
                " ) ");

        /* getting all recommendations for the selected user */
        Dataset<Row> predictions = model.transform(itemsToPredictRating);
        predictions = predictions.withColumnRenamed("prediction", "rating");

        /* joining predictions and source dataset to get recommendations with the original item names */
        return predictions
                .join(dataSet,
                        predictions.col("itemidI").equalTo(dataSet.col("itemidI")), "inner")
                .select(
                        dataSet.col("itemid"),
                        predictions.col("rating"))
                .distinct();
    }

    /**
     * getting predictions for the user with given id using contextual post-filtering
     */
    private static Dataset<Row> getPredictionsUsingPostFiltering(String userId) {
        /* getting list of the items for predict */
        Integer userIdI = Arrays.asList(userIndexer.labels()).indexOf(userId);
        dataSet.createOrReplaceTempView("dataset");
        Dataset<Row> itemsToPredictRating = spark.sql("select distinct " + userIdI + " useridI, itemidI " +
                " from dataset " +
                " where itemidI not in ( " +
                " select itemidI from dataset where userIdI = " + userIdI + " " +
                " ) ");
        Dataset<Row> predictions = model.transform(itemsToPredictRating);
        predictions = predictions.withColumnRenamed("prediction", "rating");

        /* contextual post-filtering of the source dataset using user context */
        /* contextual post-filtering approach adjusts the obtained recommendation list for each user using
        contextual information. The recommendation list adjustments can be made by:
        • Filtering out recommendations that are irrelevant (in a given context), or
        • Adjusting the ranking of recommendations on the list (based on a given context).
        we use filtering. */
        Dataset<Row> dataSetFilteredWithContext = dataSet;
        for (Map.Entry<String, String> contextVar : userContext.entrySet()) {
            dataSetFilteredWithContext = dataSetFilteredWithContext.filter(col(contextVar.getKey()).equalTo(contextVar.getValue()));
        }

        /* joining predictions and filtered dataset to get only post-filtered recommendations with original item names */
        return predictions
                .join(dataSetFilteredWithContext,
                        predictions.col("itemidI").equalTo(dataSetFilteredWithContext.col("itemidI")), "inner")
                .select(
                        dataSetFilteredWithContext.col("itemid"),
                        predictions.col("rating"))
                .distinct();
    }

    /**
     * indexing user, item and context columns by integer values
     */
    private static void createIndexes() {
        /* indexing itemid column by integer values */
        itemIndexer = new StringIndexer()
                .setInputCol("itemid")
                .setOutputCol("itemidI")
                .fit(dataSet);
        dataSet = itemIndexer.transform(dataSet);

        /* indexing userid column by integer values */
        userIndexer = new StringIndexer()
                .setInputCol("userid")
                .setOutputCol("useridI")
                .fit(dataSet);
        dataSet = userIndexer.transform(dataSet);
    }

    /**
     * training the ALS model
     */
    private static void trainModel() {
        /* splitting dataSet into test and training splits */
        Dataset<Row>[] splits = dataSet.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row>training = splits[0];
        Dataset<Row>test = splits[1];
        Long numTraining = training.count();
        Long numTest = test.count();
        System.out.println("Training: " + numTraining + ", test: " + numTest);

        /* training the model using ALS */
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setMetricName("rmse")
                .setLabelCol("rating")
                .setPredictionCol("prediction");
        ALS als = new ALS()
                .setCheckpointInterval(2)
                .setUserCol("useridI")
                .setItemCol("itemidI")
                .setRatingCol("rating")
                .setNonnegative(true);
        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(als.regParam(), new double[]{0.01, 0.1})
                .addGrid(als.maxIter(), new int[]{1, 10})
                .addGrid(als.rank(), new int[]{8, 12})
                .build();
        TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
                .setEstimator(als)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(paramGrid)
                .setTrainRatio(0.8);
        TrainValidationSplitModel trainValidationSplitModel = trainValidationSplit.fit(training);
        model = (ALSModel) trainValidationSplitModel.bestModel();

        /* evaluating the best model on the test set */
        Dataset<Row> predictions = model.transform(test);
        double RMSE = evaluator.evaluate(predictions);

        /* creating a naive baseline and compare it with the best model */
        Double meanRating = training.select(avg("rating")).head().getDouble(0);
        Double baselineRMSE = Math.sqrt(test.select(avg(col("rating").$minus(meanRating).multiply(col("rating").$minus(meanRating)))).head().getDouble(0));
        Double improvement = (baselineRMSE - RMSE) / baselineRMSE * 100;
        System.out.println(String.format("model mean Rating %.3f baseline RMSE %.3f model RMSE %.3f", meanRating, baselineRMSE, RMSE));
        System.out.println(String.format("The model differs from the baseline by %.3f percents", improvement));

        /*alsModel.save("alsModel");
        alsModel = ALSModel.load("alsModel");*/
    }
}

