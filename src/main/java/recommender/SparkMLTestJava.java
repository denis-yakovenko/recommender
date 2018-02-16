package recommender;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;
import scala.collection.TraversableOnce;
import scala.collection.immutable.ListSet;
import scala.runtime.AbstractFunction1;

import java.io.File;
import java.sql.Timestamp;
import java.util.*;

import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.sum;
import static org.apache.spark.sql.types.DataTypes.IntegerType;

public class SparkMLTestJava {
    public static void main(String[] args) {

        SparkSession spark = SparkSession.builder().master("local[*]").appName("CARS").getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");

        /** selecting data set */
        Dataset<Row> dataSet = getDePaulMovieDataSet(spark);
        //Dataset<Row> dataSet = getML100KDataSet(spark);
        dataSet.show(false);
        dataSet.printSchema();

        /** grouping dataset by computing total rating of different events using weight and amount of each item */
        dataSet = dataSet.groupBy(
                col("userid"),
                col("itemid"),
                col("Time"),
                col("Location"),
                col("Companion")
        ).agg(
                sum(
                        col("Weight").multiply(col("Value"))
                ).alias("rating")
        ).toDF();
        dataSet.show(false);
        dataSet.printSchema();

        //spark.stop();
        //System.exit(0);
        //dataSet.printSchema();
        //dataSet.show(20, false);

        /** indexing itemid column by integer values */
        StringIndexerModel itemIndexer = new StringIndexer()
                .setInputCol("itemid")
                .setOutputCol("itemidIndexed")
                .fit(dataSet);
        dataSet = itemIndexer.transform(dataSet);

        /** indexing userid column by integer values */
        StringIndexerModel userIndexer = new StringIndexer()
                .setInputCol("userid")
                .setOutputCol("useridIndexed")
                .fit(dataSet);
        dataSet = userIndexer.transform(dataSet);
        dataSet.cache();
        dataSet.show(false);
        dataSet.printSchema();

        /** contextual pre-filtering */
        Map<String, String> userContext = new HashMap<>();
        userContext.put("Time", "Weekend");
        //userContext.put("Location", "Home");
        //userContext.put("Companion", "Family");
        for (Map.Entry<String, String> contextVar : userContext.entrySet()) {
            dataSet = dataSet.filter(col(contextVar.getKey()).equalTo(contextVar.getValue()));
        }

        /** showing dataSet total counts */
        Long numRatings = dataSet.count();
        Long numUsers = dataSet.select("useridIndexed").distinct().count();
        Long numMovies = dataSet.select("itemidIndexed").distinct().count();
        System.out.println("Got " + numRatings + " ratings from " + numUsers + " users on " + numMovies + " movies.");

        /** splitting dataSet into test and training splits */
        Dataset<Row>[] splits = dataSet.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];
        Long numTraining = training.count();
        Long numTest = test.count();
        System.out.println("Training: " + numTraining + ", test: " + numTest);

        /** train model */
        ALS als = new ALS()
                .setCheckpointInterval(2)
                .setUserCol("useridIndexed")
                .setItemCol("itemidIndexed")
                .setRatingCol("rating");
        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(als.regParam(), new double[]{0.01, 0.1})
                .addGrid(als.maxIter(), new int[]{1, 10})
                .addGrid(als.rank(), new int[]{8, 12})
                .build();
        RegressionEvaluator alsEvaluator = new RegressionEvaluator()
                .setMetricName("rmse")
                .setLabelCol("rating")
                .setPredictionCol("prediction");
        TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
                .setEstimator(als)
                .setEvaluator(alsEvaluator)
                .setEstimatorParamMaps(paramGrid)
                .setTrainRatio(0.8);
        TrainValidationSplitModel model = trainValidationSplit.fit(training);
        ALSModel bestModel = (ALSModel) model.bestModel();

        /*bestModel.userFactors().show(false);
        bestModel.itemFactors().show(false);*/

        /** evaluate the best model on the test set */
        Dataset<Row> alsPredictions = bestModel.transform(test);
        double RMSE = alsEvaluator.evaluate(alsPredictions);
        System.out.println("Best model RMSE = " + RMSE);

        /** create a naive baseline and compare it with the best model */
        Double meanRating = training.select(avg("rating")).head().getDouble(0);
        Double baselineRMSE = Math.sqrt(test.select(avg(col("rating").$minus(meanRating).multiply(col("rating").$minus(meanRating)))).head().getDouble(0));
        Double improvement = (baselineRMSE - RMSE) / baselineRMSE * 100;
        System.out.println("meanRating " + meanRating + " baselineRMSE " + baselineRMSE + " best model RMSE " + RMSE);
        System.out.println("The best model improves the baseline by " + String.format("%1.2f", improvement) + "%.");

        /*IndexToString itemBackIndexer = new IndexToString()
                .setInputCol("itemid")
                .setOutputCol("itemidIndexed")
                .setLabels(itemIndexer.labels());
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{itemBackIndexer});
        PipelineModel model = pipeline.fit(training);
        Dataset<Row> predictions = model.transform(test);
        predictions.select("itemidIndexed", "itemid").show(5);*/

        /** get recommendations for all users */
        Dataset<Row> recommendForAllUsers = bestModel.recommendForAllUsers(5);
        Dataset<Row> recommendForAllItems = bestModel.recommendForAllItems(5);
        recommendForAllUsers.show(false);
        recommendForAllItems.show(false);

        spark.stop();
    }

    static Dataset<Row> getDePaulMovieDataSet(SparkSession spark) {
        return spark
                .read()
                .option("header", true)
                .option("inferSchema", true)
                .csv(new File("Movie_DePaulMovie", "ratings.txt").toString());
    }

    static Dataset<Row> getML100KDataSet(SparkSession spark) {
        Dataset<Row> dataSet = spark
                .read()
                .csv(new File("data", "ratings.csv").toString());
        dataSet = dataSet
                .withColumnRenamed("_c0", "userid")
                .withColumnRenamed("_c1", "itemid")
                .withColumnRenamed("_c2", "rating");
        dataSet = dataSet
                .withColumn("userid", dataSet.col("userid").cast(IntegerType))
                .withColumn("itemid", dataSet.col("itemid").cast(IntegerType))
                .withColumn("rating", dataSet.col("rating").cast(IntegerType));
        return dataSet;
        /*dataSet.registerTempTable("dataSet");
        return spark.sql("select cast(_c0 as int) as userid, cast(_c1 as int) as itemid, cast(_c2 as double) as rating from dataset");*/
    }
}

