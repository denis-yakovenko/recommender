package recommender;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.ml.feature.*;
import org.apache.spark.mllib.regression.FMModel;
import org.apache.spark.mllib.regression.FMWithLBFGS;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.*;
import scala.Tuple2;
import scala.Tuple3;
import scala.collection.JavaConverters;
import scala.collection.Seq;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

import static org.apache.spark.sql.functions.*;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.desc;
import static org.apache.spark.sql.functions.round;
import static org.apache.spark.sql.types.DataTypes.*;
import static org.apache.spark.sql.types.DataTypes.DoubleType;
import static org.apache.spark.sql.types.DataTypes.createStructField;

public class ContextualModeling {
    private static Dataset<Row> dataSet;
    private static SparkSession spark;
    private static String[] contextDimensions = new String[]{"Time", "Location", "Companion"};
    private static Map<String, String> userContext = new HashMap<>();
    private static RDD<LabeledPoint> convertedToLabeledPoint;
    private static FMModel model;
    private static int FeaturesCount;

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
        showStatistics();

        /* rescaling rating to range [0, 1] */
        rescaleRatings();

        /* pivoting values to columns */
        pivotRowsToColumns();

        /* converting data to LabeledPoint to train model */
        convertToLabeledPoint();

        /* training the Factorization Machine Regression model given an RDD of (label, features) pairs */
        trainModel();

        /* creating an example of the user context */
        userContext.put("Time", "Weekend");
        //userContext.put("Location", "Home");
        userContext.put("Companion", "Partner");

        /* getting and showing predictions for the user with id 1001 using contextual modeling */
        Dataset<Row> predictions = getPredictions("1001");
        predictions
                .orderBy(desc("rating"))
                .show((int) predictions.count(), false);

        spark.stop();
    }

    /**
     * getting predictions for the user with given id
     * returns an RDD of (itemId, rating) pairs
     */
    private static Dataset<Row> getPredictions(String userId) {
        List<String> columns = new ArrayList<>(Arrays.asList(dataSet.columns()));
        columns.removeAll(Arrays.asList(
                "userid",
                "itemid",
                "rating"));
        columns.removeAll(Arrays.asList(contextDimensions));
        List<Integer> featureIndexes = new ArrayList<>();
        featureIndexes.add(columns.indexOf("user_" + userId));
        featureIndexes.addAll(
                userContext.entrySet().stream().map(
                        contextVariable -> columns.indexOf(
                                contextVariable.getKey() + "_" + contextVariable.getValue()
                        )
                ).collect(Collectors.toList())
        );
        /* creating list of the vectors to predict
        * each vector has persistent user and context columns,
        * and variable item columns values
        * */
        double[] vectorValues = new double[featureIndexes.size() + 1];
        Arrays.fill(vectorValues, 1);
        Integer[] vectorIndexes = featureIndexes.toArray(new Integer[0]);
        List<Row> predictRows = new ArrayList<>();
        for (Row itemRow : dataSet.select(col("itemid")).distinct().collectAsList()) {
            String itemId = itemRow.getAs(0);
            List<Integer> vectorIndexesForEachItem = new ArrayList<>(Arrays.asList(vectorIndexes));
            vectorIndexesForEachItem.add(columns.indexOf("item_" + itemId));
            Vector features = Vectors.sparse(
                    FeaturesCount,
                    vectorIndexesForEachItem.stream().mapToInt(i -> i).toArray(),
                    vectorValues);
            Double rating = model.predict(features);
            predictRows.add(RowFactory.create(itemId, rating));
        }
        StructType schema = createStructType(new StructField[]{
                createStructField("itemid", StringType, false),
                createStructField("rating", DoubleType, false)
        });
        return spark.createDataFrame(predictRows, schema);
    }

    /**
     * training the Factorization Machine Regression model given an RDD of (label, features) pairs
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

        JavaPairRDD<Double, Double> predictions = testLP.toJavaRDD().mapToPair(
                p -> {
                    double predict = model.predict(p.features());
                    return new Tuple2<>(predict, p.label());
                }
        );
        /*System.out.println("validate: real rating => predicted rating");
        for (Tuple2<Double, Double> t : predictions.collect())
            System.out.println(
                    String.format("real %.3f => predicted %.3f delta %.3f", t._2, t._1, Math.abs(t._1 - t._2)));*/
        Double RMSE = Math.sqrt(predictions.mapToDouble(v -> Math.pow(v._1() - v._2(), 2)).mean());
        Double meanRating = training.toJavaRDD().mapToDouble(LabeledPoint::label).mean();
        Double baselineRMSE = Math.sqrt(
                testLP.toJavaRDD().mapToDouble(
                        p -> Math.pow(p.label() - meanRating, 2)
                ).mean()
        );
        System.out.println(String.format(
                "model mean Rating %.3f baseline RMSE %.3f model RMSE %.3f", meanRating, baselineRMSE, RMSE));
        Double improvement = (baselineRMSE - RMSE) / baselineRMSE * 100;
        System.out.println(String.format(
                "The model differs from the baseline by %.3f percents", improvement));

        StructType schema = createStructType(new StructField[]{
                createStructField("prediction", DataTypes.DoubleType, false),
                createStructField("label", DataTypes.DoubleType, false)
        });

        Dataset<Row> predictionsDF = spark
                .sqlContext()
                .createDataFrame(
                        predictions.map(
                                tuple -> RowFactory.create(tuple._1(), tuple._2())
                        ), schema)
                .toDF().select(
                        round(col("label")/*, 2*/)
                                .cast("double")
                                .as("label"),
                        round(col("prediction")/*, 2*/)
                                .cast("double")
                                .as("prediction")
                );
        /*predictionsDF.orderBy(
                desc("prediction"),
                desc("label")
        )
                .show((int) predictionsDF.count(), false);*/

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictionsDF);
        System.out.println("Test set accuracy = " + accuracy);

    }

    /**
     * converting data to LabeledPoint to train model
     */
    private static void convertToLabeledPoint() {
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
     * pivoting values to columns
     */
    private static void pivotRowsToColumns() {
        /* pivoting user values to columns */
        Dataset<Row> dataSetPivotedByUser = dataSet
                .groupBy(col("userid"), col("itemid"), col("rating"), col("Time"), col("Location"), col("Companion"))
                .pivot("userid")
                .agg(lit(1));
        dataSetPivotedByUser = dataSetPivotedByUser.na().fill(0.0);
        for (Object c : dataSet.select("userid").distinct().toJavaRDD().map(r -> r.getAs("userid")).collect()) {
            dataSetPivotedByUser = dataSetPivotedByUser.withColumnRenamed(c.toString(), "user_" + c);
            dataSetPivotedByUser = dataSetPivotedByUser.withColumn("user_" + c, col("user_" + c).cast("double"));
        }

        /* pivoting item values to columns */
        Dataset<Row> dataSetPivotedByItem = dataSet
                .groupBy(col("userid"), col("itemid"), col("rating"), col("Time"), col("Location"), col("Companion"))
                .pivot("itemid")
                .agg(lit(1));
        dataSetPivotedByItem = dataSetPivotedByItem.na().fill(0.0);
        for (Object c : dataSet.select("itemid").distinct().toJavaRDD().map(r -> r.getAs("itemid")).collect()) {
            dataSetPivotedByItem = dataSetPivotedByItem.withColumnRenamed(c.toString(), "item_" + c);
        }

        /* pivoting Time values to columns */
        Dataset<Row> dataSetPivotedByTime = dataSet
                .groupBy(col("userid"), col("itemid"), col("rating"), col("Time"), col("Location"), col("Companion"))
                .pivot("Time")
                .agg(lit(1));
        dataSetPivotedByTime = dataSetPivotedByTime.na().fill(0.0);
        for (Object c : dataSet.select("Time").distinct().toJavaRDD().map(r -> r.getAs("Time")).collect()) {
            dataSetPivotedByTime = dataSetPivotedByTime.withColumnRenamed(c.toString(), "Time_" + c);
        }

        /* pivoting Location values to columns */
        Dataset<Row> dataSetPivotedByLocation = dataSet
                .groupBy(col("userid"), col("itemid"), col("rating"), col("Time"), col("Location"), col("Companion"))
                .pivot("Location")
                .agg(lit(1));
        dataSetPivotedByLocation = dataSetPivotedByLocation.na().fill(0.0);
        for (Object c : dataSet.select("Location").distinct().toJavaRDD().map(r -> r.getAs("Location")).collect()) {
            dataSetPivotedByLocation = dataSetPivotedByLocation.withColumnRenamed(c.toString(), "Location_" + c);
        }

        /* pivoting Companion values to columns */
        Dataset<Row> dataSetPivotedByCompanion = dataSet
                .groupBy(col("userid"), col("itemid"), col("rating"), col("Time"), col("Location"), col("Companion"))
                .pivot("Companion")
                .agg(lit(1));
        dataSetPivotedByCompanion = dataSetPivotedByCompanion.na().fill(0.0);
        for (Object c : dataSet.select("Companion").distinct().toJavaRDD().map(r -> r.getAs("Companion")).collect()) {
            dataSetPivotedByCompanion = dataSetPivotedByCompanion.withColumnRenamed(c.toString(), "Companion_" + c);
        }

        /* joining pivoted data sets into one */
        Seq<String> keyFields = JavaConverters.asScalaIteratorConverter(
                Arrays.asList("userid", "itemid", "rating", "Time", "Location", "Companion").iterator())
                .asScala()
                .toSeq();
        Dataset<Row> dataSetPivoted = dataSetPivotedByUser
                .join(dataSetPivotedByItem, keyFields)
                .join(dataSetPivotedByTime, keyFields)
                .join(dataSetPivotedByLocation, keyFields)
                .join(dataSetPivotedByCompanion, keyFields);

        /* getting list of columns to transform to vectors */
        List<String> columnsList = new ArrayList<>(Arrays.asList(dataSetPivoted.columns()));
        columnsList.removeAll(Arrays.asList(
                "userid",
                "itemid",
                "rating"));
        columnsList.removeAll(Arrays.asList(contextDimensions));

        FeaturesCount = columnsList.size();

        /* transform columns with features to vectors */
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(columnsList.toArray(new String[0]))
                .setOutputCol("features");
        dataSet = assembler.transform(dataSetPivoted);
    }

    /**
     * rescaling each feature to range [0, 1]
     */
    private static void rescaleFeatures() {
        MinMaxScaler minMaxScaler = new MinMaxScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures");
        MinMaxScalerModel minMaxScalerModel = minMaxScaler.fit(dataSet);
        dataSet = minMaxScalerModel
                .transform(dataSet)
                .drop("features")
                .withColumnRenamed("scaledFeatures", "features");
    }

    /**
     * rescaling rating to range [0, 1]
     */
    static void rescaleRatings() {
        Row rowMinMax = dataSet.agg(
                min(col("rating")),
                max(col("rating"))).head();
        Double minRating = rowMinMax.getAs(0);
        Double maxRating = rowMinMax.getAs(1);
        dataSet = dataSet
                .withColumn(
                        "rating",
                        (col("rating").minus(minRating)).divide(maxRating - minRating)
                );
    }

    /**
     * showing dataSet total counts
     */
    static void showStatistics() {
        /* showing dataSet total counts */
        Long numRatings = dataSet.count();
        Long numUsers = dataSet.select("userid").distinct().count();
        Long numMovies = dataSet.select("itemid").distinct().count();
        System.out.println("Got " + numRatings + " ratings from " + numUsers + " users on " + numMovies + " movies.");
    }
}

