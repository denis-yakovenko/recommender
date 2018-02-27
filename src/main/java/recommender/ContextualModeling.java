package recommender;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.ml.feature.*;
import org.apache.spark.mllib.regression.FMModel;
import org.apache.spark.mllib.regression.FMWithLBFGS;
import org.apache.spark.mllib.regression.FMWithSGD;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;
import scala.Tuple2;
import scala.Tuple3;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.apache.spark.sql.types.DataTypes.*;
import static org.apache.spark.sql.types.DataTypes.DoubleType;
import static org.apache.spark.sql.types.DataTypes.createStructField;

public class ContextualModeling {
    private static Dataset<Row> dataSet;
    private static StringIndexerModel itemIndexer;
    private static StringIndexerModel userIndexer;
    private static StringIndexerModel timeIndexer;
    private static StringIndexerModel locationIndexer;
    private static StringIndexerModel companionIndexer;
    private static SparkSession spark;
    private static Map<String, String> userContext = new HashMap<>();
    private static RDD<LabeledPoint> convertedToLabeledPoint;
    private static FMModel model;

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

        /* showing dataSet total counts */
        Util.showDataSetStatistics(dataSet);

        /* rescaling rating to range [0, 1] */
        dataSet = Util.getWithRescaledRating(dataSet);

        /* indexing user, item and context columns by integer values */
        createIndexes();

        /* pivoting values to columns */
        dataSet = Util.getPivotedToColumns(dataSet);

        dataSet.show(false);
        dataSet.printSchema();

        /* transforming columns with features to vectors */
        //dataSet = Util.getVectorized(dataSet);

        /* rescaling each feature to range [0, 1] */
        dataSet = Util.getWithRescaledFeatures(dataSet);

        dataSet.show(false);
        dataSet.printSchema();

        /* converting data to LabeledPoint to train model */
        convertToLabeledPoint();

        dataSet.show(false);
        dataSet.printSchema();

        /* save data in LabeledPoint format to the LibSVM-formatted file */
        Util.saveToLibSVM(spark, convertedToLabeledPoint, "dataInLibSVM");

        trainModel();

        /* getting and showing predictions for the user with id 1001 using contextual modeling */
        RDD<Row> predictions = getPredictions("1001");
        predictions.toJavaRDD().collect().forEach(System.out::println);

        spark.stop();
    }

    /**
     * getting predictions for the user with given id
     * returns an RDD of (itemId, rating) pairs
     */
    private static RDD<Row> getPredictions(String userId) {
        // TODO: this is stub. should implement the method.
        List<Row> p = Arrays.asList(
                RowFactory.create("item1", 1.00),
                RowFactory.create("item2", 0.99),
                RowFactory.create("item3", 0.98)
        );
        StructType s = createStructType(new StructField[]{
                createStructField("itemId", StringType, false),
                createStructField("rating", DoubleType, false)
        });
        Dataset<Row> predictionsDataSet = spark.createDataFrame(p, s);
        return predictionsDataSet.rdd();
        // TODO: this is stub. should implement the method.
    }

    /**
     * training the Factoriaton Machine Regression model given an RDD of (label, features) pairs
     */
    private static void trainModel() {
        /* splitting dataSet into test and training splits */
        RDD<LabeledPoint>[] splits = convertedToLabeledPoint.randomSplit(new double[]{0.8, 0.2}, 0L);
        RDD<LabeledPoint> training = splits[0];
        RDD<LabeledPoint> testLP = splits[1];
        RDD<Vector> test = testLP.toJavaRDD().map(
                (Function<LabeledPoint, Vector>) LabeledPoint::features).rdd();
        Long numTraining = training.count();
        Long numTest = test.count();
        System.out.println("Training: " + numTraining + ", test: " + numTest);

        /*FMModel model = FMWithSGD.train(
                training, 0, 100, 0.15, 1.0,
                new Tuple3<>(true, true, 4),
                new Tuple3<>(0.0, 0.0, 0.0),
                0.1);*/

        model = FMWithLBFGS.train(
                training,
                0,
                20,
                15,
                new Tuple3<>(true, true, 4),
                new Tuple3<>(0.0, 0.0, 0.0),
                0.1);

        JavaPairRDD<Double, Double> predictionAndLabels1 = testLP.toJavaRDD().mapToPair(
                p -> {
                    double predict = model.predict(p.features());
                    return new Tuple2<>(predict, p.label());
                }
        );
        System.out.println("validate: real rating => predicted rating");
        for (Tuple2<Double, Double> t : predictionAndLabels1.collect())
            System.out.println(String.format("real %.3f => predicted %.3f delta %.3f", t._2, t._1, Math.abs(t._1 - t._2)));
        Double RMSE = Math.sqrt(predictionAndLabels1.mapToDouble(v -> Math.pow(v._1() - v._2(), 2)).mean());
        Double meanRating = training.toJavaRDD().mapToDouble(LabeledPoint::label).mean();
        Double baselineRMSE = Math.sqrt(testLP.toJavaRDD().mapToDouble(p -> Math.pow(p.label() - meanRating, 2)).mean());
        System.out.println(String.format("model mean Rating %.3f baseline RMSE %.3f model RMSE %.3f", meanRating, baselineRMSE, RMSE));
        Double improvement = (baselineRMSE - RMSE) / baselineRMSE * 100;
        System.out.println(String.format("The model differs from the baseline by %.3f percents", improvement));
    }

    /**
     * converting data to LabeledPoint to train model
     */
    private static void convertToLabeledPoint() {
        /* converting data to LabeledPoint to train model */
        convertedToLabeledPoint = dataSet
                .select("rating", "features")
                .toJavaRDD()
                .map(
                        (Function<Row, LabeledPoint>) r -> new LabeledPoint(
                                r.getAs(0),
                                Vectors.dense(
                                        ((org.apache.spark.ml.linalg.Vector)
                                                r.getAs("features")
                                        )
                                                .toArray()
                                ).toSparse()
                        )
                )
                .rdd();
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

        /* indexing Time column by integer values */
        timeIndexer = new StringIndexer()
                .setInputCol("Time")
                .setOutputCol("TimeI")
                .fit(dataSet);
        dataSet = timeIndexer.transform(dataSet);

        /* indexing Location column by integer values */
        locationIndexer = new StringIndexer()
                .setInputCol("Location")
                .setOutputCol("LocationI")
                .fit(dataSet);
        dataSet = locationIndexer.transform(dataSet);

        /* indexing CompanionI column by integer values */
        companionIndexer = new StringIndexer()
                .setInputCol("Companion")
                .setOutputCol("CompanionI")
                .fit(dataSet);
        dataSet = companionIndexer.transform(dataSet);
    }
}

