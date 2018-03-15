package recommender.contextualModeling;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.ml.param.*;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.ml.feature.*;
import org.apache.spark.mllib.regression.FMModel;
import org.apache.spark.mllib.regression.FMWithLBFGS;
import org.apache.spark.mllib.util.MLUtils;
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
import static org.apache.spark.sql.types.DataTypes.*;
import static org.apache.spark.sql.types.DataTypes.DoubleType;
import static org.apache.spark.sql.types.DataTypes.createStructField;

public class ContextualModeling {
    static String[] contextDimensions = new String[]{"Time", "Location", "Companion"};
    static String userIdColumnName = "userid";
    static String itemIdColumnName = "itemid";
    static String ratingColumnName = "rating";

    /**
     * getting predictions for the user with given id
     * returns an RDD of (itemId, rating) pairs
     */
    public static Dataset<Row> getPredictions(SparkSession spark,
                                              Dataset<Row> dataSet,
                                              FMModel model,
                                              Map<String, String> userContext,
                                              String userId) {
        List<String> columns = new ArrayList<>(Arrays.asList(dataSet.columns()));
        columns.removeAll(Arrays.asList(
                //uniqueIdColumnName,
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

        int featuresCount = ((SparseVector) dataSet.head().getAs("features")).size();

        List<Row> userRows = dataSet.select(col(itemIdColumnName)).distinct().collectAsList();
        for (Row itemRow : userRows) {
            String itemId = itemRow.getAs(0);
            List<Integer> vectorIndexesForEachItem = new ArrayList<>(Arrays.asList(vectorIndexes));
            vectorIndexesForEachItem.add(columns.indexOf(itemIdColumnName + "_" + itemId));
            Vector features = Vectors.sparse(
                    featuresCount,
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
     * designing the Factorization Machine regression model
     * given an RDD of (label, features) pairs
     * using cross-validation
     */
    public static Properties designModel(RDD<LabeledPoint> convertedToLabeledPoint, int folds) {
        // splitting dataSet into folds
        double[] foldWeights = new double[folds];
        Arrays.fill(foldWeights, 1.0);
        RDD<LabeledPoint>[] foldSplits = convertedToLabeledPoint.randomSplit(foldWeights, 0L);

        Properties bestModelParams = new Properties();
        Double bestRMSE = Double.MAX_VALUE;
        ParamGridBuilder paramGridBuilder = new ParamGridBuilder();
        paramGridBuilder.addGrid(new IntParam("", "iterations", ""), new int[]{100});
        paramGridBuilder.addGrid(new BooleanParam("", "globalBiasTerm", ""));
        paramGridBuilder.addGrid(new BooleanParam("", "oneWayInteraction", ""));
        paramGridBuilder.addGrid(new IntParam("", "pairwiseInteractionsFactors", ""), new int[]{8});
        paramGridBuilder.addGrid(new DoubleParam("", "regIntercept", ""), new double[]{0.0, 0.1, 1.0});
        paramGridBuilder.addGrid(new DoubleParam("", "regOneWayInteractions", ""), new double[]{0.0, 0.1, 1.0});
        paramGridBuilder.addGrid(new DoubleParam("", "regPairwiseInteractions", ""), new double[]{0.0, 0.1, 1.0});
        paramGridBuilder.addGrid(new DoubleParam("", "initStd", ""), new double[]{0.01, 0.1, 1.0});
        ParamMap[] paramMaps = paramGridBuilder.build();
        // training for each fold
        for (int f = 0; f < foldSplits.length; f++/*RDD<LabeledPoint> foldSplit : foldSplits*/) {
            RDD<LabeledPoint> foldSplit = foldSplits[f];
            System.out.println("fold " + f + " size " + foldSplit.count());
            // splitting dataSet into validation and training splits
            RDD<LabeledPoint>[] splits = foldSplit.randomSplit(new double[]{0.8, 0.2}, 0L);
            RDD<LabeledPoint> training = splits[0];
            RDD<LabeledPoint> validation = splits[1];
            System.out.println("training size " + training.count());
            System.out.println("validation size " + validation.count());
            Double meanRating = training.toJavaRDD().mapToDouble(LabeledPoint::label).mean();
            Double baselineRMSE = Math.sqrt(
                    validation.toJavaRDD().mapToDouble(
                            p -> Math.pow(p.label() - meanRating, 2)
                    ).mean()
            );
            System.out.println(String.format("Model mean Rating %.3f baseline RMSE %.3f", meanRating, baselineRMSE));
            for (ParamMap i : paramMaps) {
                Integer iterations = (Integer) i.get(new IntParam("", "iterations", "")).get();
                Boolean globalBiasTerm = (Boolean) i.get(new IntParam("", "globalBiasTerm", "")).get();
                Boolean oneWayInteraction = (Boolean) i.get(new IntParam("", "oneWayInteraction", "")).get();
                Integer pairwiseInteractionsFactors = (Integer) i.get(new IntParam("", "pairwiseInteractionsFactors", "")).get();
                Double regIntercept = (Double) i.get(new IntParam("", "regIntercept", "")).get();
                Double regOneWayInteractions = (Double) i.get(new IntParam("", "regOneWayInteractions", "")).get();
                Double regPairwiseInteractions = (Double) i.get(new IntParam("", "regPairwiseInteractions", "")).get();
                Double initStd = (Double) i.get(new IntParam("", "initStd", "")).get();

                FMModel trainingModel = FMWithLBFGS.train(
                        training,
                        0, //task 0 for Regression
                        iterations,
                        15, //not used
                        new Tuple3<>(globalBiasTerm, oneWayInteraction, pairwiseInteractionsFactors),
                        new Tuple3<>(regIntercept, regOneWayInteractions, regPairwiseInteractions),
                        initStd);

                JavaPairRDD<Double, Double> predictions = validation.toJavaRDD().mapToPair(
                        p -> {
                            double predict = trainingModel.predict(p.features());
                            return new Tuple2<>(predict, p.label());
                        }
                );

                Double RMSE = Math.sqrt(predictions.mapToDouble(v -> Math.pow(v._1() - v._2(), 2)).mean());

                Double improvement = (baselineRMSE - RMSE) / baselineRMSE * 100;
                Properties modelParams = new Properties();
                modelParams.setProperty("iterations", String.valueOf(iterations));
                modelParams.setProperty("globalBiasTerm", String.valueOf(globalBiasTerm));
                modelParams.setProperty("oneWayInteraction", String.valueOf(oneWayInteraction));
                modelParams.setProperty("pairwiseInteractionsFactors", String.valueOf(pairwiseInteractionsFactors));
                modelParams.setProperty("regIntercept", String.valueOf(regIntercept));
                modelParams.setProperty("regOneWayInteractions", String.valueOf(regOneWayInteractions));
                modelParams.setProperty("regPairwiseInteractions", String.valueOf(regPairwiseInteractions));
                modelParams.setProperty("initStd", String.valueOf(initStd));
                System.out.println(
                        modelParams.toString() + " "
                                + String.format("Model RMSE %.3f. The model improves the baseline by %.3f percents",
                                RMSE,
                                improvement));

                /*StructType schema = createStructType(new StructField[]{
                        createStructField("prediction", DataTypes.DoubleType, false),
                        createStructField("label", DataTypes.DoubleType, false)
                });
                Dataset<Row> predictionsDF = spark
                        .sqlContext()
                        .createDataFrame(predictions.map(tuple -> RowFactory.create(tuple._1(), tuple._2())), schema)
                        .toDF().select(
                                round(col("label"), 1).cast("double").as("label"),
                                round(col("prediction"), 1).cast("double").as("prediction")
                        );
                MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                        .setLabelCol("label")
                        .setPredictionCol("prediction")
                        .setMetricName("accuracy");
                double accuracy = evaluator.evaluate(predictionsDF);*/

                if (RMSE < bestRMSE) {
                    bestRMSE = RMSE;
                    bestModelParams.setProperty("iterations", String.valueOf(iterations));
                    bestModelParams.setProperty("globalBiasTerm", String.valueOf(globalBiasTerm));
                    bestModelParams.setProperty("oneWayInteraction", String.valueOf(oneWayInteraction));
                    bestModelParams.setProperty("pairwiseInteractionsFactors", String.valueOf(pairwiseInteractionsFactors));
                    bestModelParams.setProperty("regIntercept", String.valueOf(regIntercept));
                    bestModelParams.setProperty("regOneWayInteractions", String.valueOf(regOneWayInteractions));
                    bestModelParams.setProperty("regPairwiseInteractions", String.valueOf(regPairwiseInteractions));
                    bestModelParams.setProperty("initStd", String.valueOf(initStd));
                }
            }
        }
        return bestModelParams;
    }

    /**
     * training the Factorization Machine regression model
     * given an RDD of (label, features) pairs
     */
    public static FMModel trainModel(RDD<LabeledPoint> convertedToLabeledPoint, Properties modelParams) {
        Integer iterations = Integer.valueOf(modelParams.getProperty("iterations"));
        Boolean globalBiasTerm = Boolean.getBoolean(modelParams.getProperty("globalBiasTerm"));
        Boolean oneWayInteraction = Boolean.getBoolean(modelParams.getProperty("oneWayInteraction"));
        Integer pairwiseInteractionsFactors = Integer.valueOf(modelParams.getProperty("pairwiseInteractionsFactors"));
        Double regIntercept = Double.valueOf(modelParams.getProperty("regIntercept"));
        Double regOneWayInteractions = Double.valueOf(modelParams.getProperty("regOneWayInteractions"));
        Double regPairwiseInteractions = Double.valueOf(modelParams.getProperty("regPairwiseInteractions"));
        Double initStd = Double.valueOf(modelParams.getProperty("initStd"));

        return FMWithLBFGS.train(
                convertedToLabeledPoint,
                0, //task 0 for Regression
                iterations,
                15, //not used
                new Tuple3<>(globalBiasTerm, oneWayInteraction, pairwiseInteractionsFactors),
                new Tuple3<>(regIntercept, regOneWayInteractions, regPairwiseInteractions),
                initStd);
    }

    /**
     * converting data to LabeledPoint to train model
     */
    public static RDD<LabeledPoint> convertToLabeledPoint(Dataset<Row> dataSet) {
        return dataSet
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
    public static Dataset<Row> pivotRowsToFeatures(Dataset<Row> dataSet) {
        String uniqueIdColumnName = "uniqueID";
        Dataset<Row> result;
        ArrayList<String> columns = new ArrayList<>();
        columns.add(userIdColumnName);
        columns.add(itemIdColumnName);
        columns.addAll(Arrays.asList(contextDimensions));
        ArrayList<Dataset<Row>> pivotedDataSets = new ArrayList<>();
        result = dataSet.withColumn(uniqueIdColumnName, monotonicallyIncreasingId());
        for (String column : columns) {
            Dataset<Row> pivotedDataSet = result
                    .groupBy(uniqueIdColumnName)
                    .pivot(column)
                    .agg(lit(1));
            pivotedDataSet = pivotedDataSet.na().fill(0.0);
            for (Object c : result.select(column).distinct().toJavaRDD().map(r -> r.getAs(column)).collect()) {
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
            result = result.join(pivotedDataSet, keyFields);
        /* getting list of columns to transform to vectors */
        List<String> columnsList = new ArrayList<>(Arrays.asList(result.columns()));
        columnsList.removeAll(Arrays.asList(
                uniqueIdColumnName,
                userIdColumnName,
                itemIdColumnName,
                ratingColumnName));
        columnsList.removeAll(Arrays.asList(contextDimensions));
        //int featuresCount = columnsList.size();
        /* transform columns with features to vectors */
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(columnsList.toArray(new String[0]))
                .setOutputCol("features");
        return assembler.transform(result).drop(uniqueIdColumnName);
    }

    /**
     * rescaling rating to range [-1, 1]
     */
    public static Dataset<Row> rescaleRatings(Dataset<Row> dataSet) {
        Row rowMinMax = dataSet.agg(
                min(col(ratingColumnName)),
                max(col(ratingColumnName))).head();
        Double minRating = rowMinMax.getAs(0);
        Double maxRating = rowMinMax.getAs(1);
        return dataSet
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
    public static void showStatistics(Dataset<Row> dataSet) {
        /* showing dataSet total counts */
        Long numRatings = dataSet.count();
        Long numUsers = dataSet.select(userIdColumnName).distinct().count();
        Long numItems = dataSet.select(itemIdColumnName).distinct().count();
        System.out.println("Got " + numRatings + " ratings from " + numUsers + " users on " + numItems + " items.");
    }

    /**
     * loading data set
     */
    public static Dataset<Row> loadDataSet(SparkSession spark, String dataSetPath) {
        Dataset<Row> dataSet = spark
                .read()
                .option("header", true)
                .option("inferSchema", true)
                .csv(dataSetPath)
                .withColumn(ratingColumnName, col(ratingColumnName).cast("double"));
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
        return dataSet;
    }

    /**
     * loading data in LabeledPoint format from the LibSVM-formatted file
     */
    public static RDD<LabeledPoint> loadLibSVM(SparkSession spark, String dataInputFolder) {
        return MLUtils
                .loadLibSVMFile(
                        spark.sparkContext(),
                        dataInputFolder
                );
    }

    /**
     * getting spark session
     */
    public static SparkSession getSpark() {
        SparkSession spark = SparkSession
                .builder()
                .enableHiveSupport()
                .master("local[*]")
                .appName("CARS")
                .getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");
        return spark;
    }

    /**
     * save data in LabeledPoint format to the LibSVM-formatted file
     */
    public static void saveToLibSVM(SparkSession spark, RDD<LabeledPoint> rdd, String dataOutputFolder) throws IOException {
        /* save data in LabeledPoint format to the LibSVM-formatted file */
        FileSystem fileSystem = FileSystem.get(spark.sparkContext().hadoopConfiguration());
        Path outDir = new Path(dataOutputFolder);
        if (fileSystem.exists(outDir))
            fileSystem.delete(outDir, true);
        MLUtils.saveAsLibSVMFile(rdd.repartition(1, null), dataOutputFolder);
    }
}

