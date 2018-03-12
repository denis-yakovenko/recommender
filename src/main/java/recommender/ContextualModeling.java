package recommender;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.FMWithSGD;
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
    private static String[] contextDimensions;
    private static Map<String, String> userContext = new HashMap<>();
    private static RDD<LabeledPoint> convertedToLabeledPoint;
    private static FMModel model;
    private static int FeaturesCount;
    private static String uniqueIdColumnName = "uniqueID";
    private static String userIdColumnName;
    private static String itemIdColumnName;
    private static String ratingColumnName;

    public static void main(String[] args) throws IOException {
        spark = SparkSession
                .builder()
                .enableHiveSupport()
                .master("local[*]")
                .appName("CARS")
                //.config("spark.sql.warehouse.dir", "file:///c:/tmp/spark-warehouse")
                .getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");

        /* creating an example of the user context */
        userContext.put("Time", "Weekend");
        userContext.put("Location", "Home");
        userContext.put("Companion", "Partner");

        /* initialize column names */
        contextDimensions = new String[]{"Time", "Location", "Companion"};
        userIdColumnName = "userid";
        itemIdColumnName = "itemid";
        ratingColumnName = "rating";

        /* loading data set */
        loadDataSet("Movie_DePaulMovie/ratingsOriginal.txt");

        /* showing dataSet total counts */
        showStatistics();

        /* rescaling rating to range [-1, 1] */
        rescaleRatings();

        /* pivoting values to columns and transform to vectors of features */
        pivotRowsToFeatures();

        /* converting data to LabeledPoint to train model */
        convertToLabeledPoint();

        /* save data in LabeledPoint format to the LibSVM-formatted file */
        //Util.saveToLibSVM(spark, convertedToLabeledPoint, "pivotedDataSet");
        /* load data in LabeledPoint format from the LibSVM-formatted file */
        //convertedToLabeledPoint = Util.loadLibSVM(spark, "pivotedDataSet");

        for (int i = 1; i <= 5; i++) {

        /* training the Factorization Machine Regression model given an RDD of (label, features) pairs */
            trainModel();

        /* getting and showing predictions for the user with id 1001 using contextual modeling */
            Dataset<Row> predictions = getPredictions("1001");
            predictions
                    .orderBy(desc(ratingColumnName))
                    .show((int) predictions.count(), false);
        }

        spark.stop();
    }

    /**
     * getting predictions for the user with given id
     * returns an RDD of (itemId, rating) pairs
     */
    private static Dataset<Row> getPredictions(String userId) {
        List<String> columns = new ArrayList<>(Arrays.asList(dataSet.columns()));
        columns.removeAll(Arrays.asList(
                uniqueIdColumnName,
                userIdColumnName,
                itemIdColumnName,
                ratingColumnName));
        columns.removeAll(Arrays.asList(contextDimensions));
        List<Integer> featureIndexes = new ArrayList<>();
        featureIndexes.add(columns.indexOf(userIdColumnName + "_" + userId));
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
        for (Row itemRow : dataSet.select(col(itemIdColumnName)).distinct().collectAsList()) {
            String itemId = itemRow.getAs(0);
            List<Integer> vectorIndexesForEachItem = new ArrayList<>(Arrays.asList(vectorIndexes));
            vectorIndexesForEachItem.add(columns.indexOf(itemIdColumnName + "_" + itemId));
            Vector features = Vectors.sparse(
                    FeaturesCount,
                    vectorIndexesForEachItem.stream().mapToInt(i -> i).toArray(),
                    vectorValues);
            Double rating = model.predict(features);
            predictRows.add(RowFactory.create(itemId, rating));
        }
        StructType schema = createStructType(new StructField[]{
                createStructField(itemIdColumnName, StringType, false),
                createStructField(ratingColumnName, DoubleType, false)
        });
        return spark.createDataFrame(predictRows, schema).filter(col(ratingColumnName).$greater(0));
    }

    /**
     * training the Factorization Machine Regression model given an RDD of (label, features) pairs
     */
    private static void trainModel() {
        /* splitting dataSet into test and training splits */

        if (1 == 0) {
            RDD<LabeledPoint>[] splits = convertedToLabeledPoint.randomSplit(new double[]{0.8, 0.2}, 0L);
            RDD<LabeledPoint> training = splits[0];
            RDD<LabeledPoint> test = splits[1];
        }
        RDD<LabeledPoint> training = convertedToLabeledPoint;
        RDD<LabeledPoint> test = convertedToLabeledPoint;
        /*Long numTraining = training.count();
        Long numTest = test.count();
        System.out.println("Training: " + numTraining + ", test: " + numTest);*/

        /*model = FMWithSGD.train(
                training, 0, 100, 0.15, 1.0,
                new Tuple3<>(false, false, 4),
                new Tuple3<>(0.0, 0.0, 0.0),
                0.1);*/
        //if (1 == 0)
        model = FMWithLBFGS.train(
                training,
                0,
                100,
                15,
                //new Tuple3<>(true, true, 4),
                //new Tuple3<>(false, true, 4),
                //new Tuple3<>(true, false, 4),
                new Tuple3<>(false, false, 4),
                new Tuple3<>(0.0, 0.0, 0.0),
                0.1);
        if (1 == 0) {

            JavaPairRDD<Double, Double> predictions = test.toJavaRDD().mapToPair(
                    p -> {
                        double predict = model.predict(p.features());
                        return new Tuple2<>(predict, p.label());
                    }
            );
            Double RMSE = Math.sqrt(predictions.mapToDouble(v -> Math.pow(v._1() - v._2(), 2)).mean());
            Double meanRating = training.toJavaRDD().mapToDouble(LabeledPoint::label).mean();
            Double baselineRMSE = Math.sqrt(
                    test.toJavaRDD().mapToDouble(
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

            MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                    .setLabelCol("label")
                    .setPredictionCol("prediction")
                    .setMetricName("accuracy");
            double accuracy = evaluator.evaluate(predictionsDF);
            System.out.println("Test set accuracy = " + accuracy);
        }
    }

    /**
     * converting data to LabeledPoint to train model
     */
    private static void convertToLabeledPoint() {
        convertedToLabeledPoint = dataSet
                .select(ratingColumnName, "features")
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
     * pivoting values to columns and transform to vectors of features
     */
    private static void pivotRowsToFeatures() {
        ArrayList<String> columns = new ArrayList<>();
        columns.add(userIdColumnName);
        columns.add(itemIdColumnName);
        columns.addAll(Arrays.asList(contextDimensions));
        ArrayList<Dataset<Row>> pivotedDataSets = new ArrayList<>();
        dataSet = dataSet.withColumn(uniqueIdColumnName, monotonicallyIncreasingId());
        for (String column : columns) {
            Dataset<Row> pivotedDataSet = dataSet
                    .groupBy(uniqueIdColumnName)
                    .pivot(column)
                    .agg(lit(1));
            pivotedDataSet = pivotedDataSet.na().fill(0.0);
            for (Object c : dataSet.select(column).distinct().toJavaRDD().map(r -> r.getAs(column)).collect()) {
                pivotedDataSet = pivotedDataSet.withColumnRenamed(c.toString(), column + "_" + c);
            }
            pivotedDataSets.add(pivotedDataSet);
        }
        Seq<String> keyFields = JavaConverters.asScalaIteratorConverter(
                Collections.singletonList(uniqueIdColumnName).iterator())
                .asScala()
                .toSeq();
        /* joining pivoted data sets into one */
        for (Dataset<Row> pivotedDataSet : pivotedDataSets)
            dataSet = dataSet.join(pivotedDataSet, keyFields);
        /* getting list of columns to transform to vectors */
        List<String> columnsList = new ArrayList<>(Arrays.asList(dataSet.columns()));
        columnsList.removeAll(Arrays.asList(
                uniqueIdColumnName,
                userIdColumnName,
                itemIdColumnName,
                ratingColumnName));
        columnsList.removeAll(Arrays.asList(contextDimensions));
        FeaturesCount = columnsList.size();
        /* transform columns with features to vectors */
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(columnsList.toArray(new String[0]))
                .setOutputCol("features");
        dataSet = assembler.transform(dataSet);
    }

    /**
     * rescaling rating to range [-1, 1]
     */
    private static void rescaleRatings() {
        Row rowMinMax = dataSet.agg(
                min(col(ratingColumnName)),
                max(col(ratingColumnName))).head();
        Double minRating = rowMinMax.getAs(0);
        Double maxRating = rowMinMax.getAs(1);
        dataSet = dataSet
                .withColumn(
                        ratingColumnName,
                        (col(ratingColumnName).minus(minRating))
                                .divide(maxRating - minRating)
                                .multiply(2)
                                .minus(1)
                );
    }

    /**
     * showing dataSet total counts
     */
    private static void showStatistics() {
        /* showing dataSet total counts */
        Long numRatings = dataSet.count();
        Long numUsers = dataSet.select(userIdColumnName).distinct().count();
        Long numItems = dataSet.select(itemIdColumnName).distinct().count();
        System.out.println("Got " + numRatings + " ratings from " + numUsers + " users on " + numItems + " items.");
    }

    /**
     * loading data set
     */
    private static void loadDataSet(String dataSetPath) {
        dataSet = spark
                .read()
                .option("header", true)
                .option("inferSchema", true)
                .csv(dataSetPath)
                .withColumn(ratingColumnName, col(ratingColumnName).cast("double")).toDF();
        List<String> columnsList = new ArrayList<>();
        columnsList.addAll(Arrays.asList(
                userIdColumnName,
                itemIdColumnName,
                ratingColumnName));
        Collections.addAll(columnsList, contextDimensions);
        for (String column : dataSet.columns())
            if (!columnsList.contains(column))
                dataSet = dataSet.drop(column);
        dataSet.cache();
    }
}

