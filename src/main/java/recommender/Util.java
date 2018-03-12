package recommender;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.ml.feature.*;
import org.apache.spark.mllib.regression.FMModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.collection.JavaConverters;
import scala.collection.Seq;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.apache.spark.sql.functions.*;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.types.DataTypes.DoubleType;
import static org.apache.spark.sql.types.DataTypes.IntegerType;

class Util {

    /**
     * transforming columns with features to vectors
     */
    static Dataset<Row> getVectorized(Dataset<Row> dataSet) {
        /* transforming columns with features to vectors */
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"useridI", "itemidI", "TimeI", "LocationI", "CompanionI"})
                .setOutputCol("features");
        return assembler.transform(dataSet)
                .withColumn("label", col("rating"));
    }

    /**
     * loading data set
     */
    static Dataset<Row> getDePaulMovieDataSet(SparkSession spark) {
        Dataset<Row> dataSet = spark
                .read()
                .option("header", true)
                .option("inferSchema", true)
                .csv(new File("Movie_DePaulMovie", "ratings.txt").toString());

        /* grouping dataset by computing total rating of different events using weight and amount of each item */
        return dataSet
                .groupBy(
                        col("userid"),
                        col("itemid"),
                        col("Time"),
                        col("Location"),
                        col("Companion"))
                .agg(
                        sum(
                                col("Weight")
                                        .multiply(col("Value")))
                                .alias("rating"))
                .withColumn("rating", col("rating").cast("double"))
                .toDF();
    }

    /**
     * loading data set
     */
    static Dataset<Row> getDePaulMovieDataSetOriginal(SparkSession spark) {
        return spark
                .read()
                .option("header", true)
                .option("inferSchema", true)
                .csv(new File("Movie_DePaulMovie", "ratingsOriginal.txt").toString())
                .withColumn("rating", col("rating").cast("double")).toDF();
    }

    /**
     * loading data set
     */
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
                .withColumn("rating", dataSet.col("rating").cast(DoubleType));
        return dataSet;
        /*dataSet.registerTempTable("dataSet");
        return spark.sql("select cast(_c0 as int) as userid, cast(_c1 as int) as itemid, cast(_c2 as double) as rating from dataset");*/
    }

    /**
     * saving the model
     */
    static void saveModel(SparkSession spark, FMModel model) throws IOException {
        FileSystem fileSystem = FileSystem.get(spark.sparkContext().hadoopConfiguration());
        Path outDir = new Path("modelFMWithLBFGS");
        if (fileSystem.exists(outDir))
            fileSystem.delete(outDir, true);
        model.save(spark.sparkContext(), "modelFMWithLBFGS");
    }

    /**
     * load data in LabeledPoint format from the LibSVM-formatted file
     */
    static RDD<LabeledPoint> loadLibSVM(SparkSession spark, String dataInputFolder) {
        return MLUtils
                .loadLibSVMFile(
                        spark.sparkContext(),
                        dataInputFolder
                );
    }

    /**
     * indexing user, item and context columns by integer values
     */
    /*private static void createIndexes() {
        *//* indexing itemid column by integer values *//*
        ContextualModeling.itemIndexer = new StringIndexer()
                .setInputCol("itemid")
                .setOutputCol("itemidI")
                .fit(ContextualModeling.dataSet);
        ContextualModeling.dataSet = ContextualModeling.itemIndexer.transform(ContextualModeling.dataSet);

        *//* indexing userid column by integer values *//*
        ContextualModeling.userIndexer = new StringIndexer()
                .setInputCol("userid")
                .setOutputCol("useridI")
                .fit(ContextualModeling.dataSet);
        ContextualModeling.dataSet = ContextualModeling.userIndexer.transform(ContextualModeling.dataSet);

        *//* indexing Time column by integer values *//*
        ContextualModeling.timeIndexer = new StringIndexer()
                .setInputCol("Time")
                .setOutputCol("TimeI")
                .fit(ContextualModeling.dataSet);
        ContextualModeling.dataSet = ContextualModeling.timeIndexer.transform(ContextualModeling.dataSet);

        *//* indexing Location column by integer values *//*
        ContextualModeling.locationIndexer = new StringIndexer()
                .setInputCol("Location")
                .setOutputCol("LocationI")
                .fit(ContextualModeling.dataSet);
        ContextualModeling.dataSet = ContextualModeling.locationIndexer.transform(ContextualModeling.dataSet);

        *//* indexing CompanionI column by integer values *//*
        ContextualModeling.companionIndexer = new StringIndexer()
                .setInputCol("Companion")
                .setOutputCol("CompanionI")
                .fit(ContextualModeling.dataSet);
        ContextualModeling.dataSet = ContextualModeling.companionIndexer.transform(ContextualModeling.dataSet);
    }*/

    /**
     * pivoting values to columns
     */
    private static void pivotRowsToColumns() {
        /*Dataset<Row> dataSetPivotedByUser = dataSet
                .groupBy(col("userid"), col("itemid"), col("rating"), col("Time"), col("Location"), col("Companion"))
                .pivot("userid")
                .agg(lit(1));
        dataSetPivotedByUser = dataSetPivotedByUser.na().fill(0.0);
        for (Object c : dataSet.select("userid").distinct().toJavaRDD().map(r -> r.getAs("userid")).collect()) {
            dataSetPivotedByUser = dataSetPivotedByUser.withColumnRenamed(c.toString(), "user_" + c);
            dataSetPivotedByUser = dataSetPivotedByUser.withColumn("user_" + c, col("user_" + c).cast("double"));
        }

        Dataset<Row> dataSetPivotedByItem = dataSet
                .groupBy(col("userid"), col("itemid"), col("rating"), col("Time"), col("Location"), col("Companion"))
                .pivot("itemid")
                .agg(lit(1));
        dataSetPivotedByItem = dataSetPivotedByItem.na().fill(0.0);
        for (Object c : dataSet.select("itemid").distinct().toJavaRDD().map(r -> r.getAs("itemid")).collect()) {
            dataSetPivotedByItem = dataSetPivotedByItem.withColumnRenamed(c.toString(), "item_" + c);
        }

        Dataset<Row> dataSetPivotedByTime = dataSet
                .groupBy(col("userid"), col("itemid"), col("rating"), col("Time"), col("Location"), col("Companion"))
                .pivot("Time")
                .agg(lit(1));
        dataSetPivotedByTime = dataSetPivotedByTime.na().fill(0.0);
        for (Object c : dataSet.select("Time").distinct().toJavaRDD().map(r -> r.getAs("Time")).collect()) {
            dataSetPivotedByTime = dataSetPivotedByTime.withColumnRenamed(c.toString(), "Time_" + c);
        }

        Dataset<Row> dataSetPivotedByLocation = dataSet
                .groupBy(col("userid"), col("itemid"), col("rating"), col("Time"), col("Location"), col("Companion"))
                .pivot("Location")
                .agg(lit(1));
        dataSetPivotedByLocation = dataSetPivotedByLocation.na().fill(0.0);
        for (Object c : dataSet.select("Location").distinct().toJavaRDD().map(r -> r.getAs("Location")).collect()) {
            dataSetPivotedByLocation = dataSetPivotedByLocation.withColumnRenamed(c.toString(), "Location_" + c);
        }

        Dataset<Row> dataSetPivotedByCompanion = dataSet
                .groupBy(col("userid"), col("itemid"), col("rating"), col("Time"), col("Location"), col("Companion"))
                .pivot("Companion")
                .agg(lit(1));
        dataSetPivotedByCompanion = dataSetPivotedByCompanion.na().fill(0.0);
        for (Object c : dataSet.select("Companion").distinct().toJavaRDD().map(r -> r.getAs("Companion")).collect()) {
            dataSetPivotedByCompanion = dataSetPivotedByCompanion.withColumnRenamed(c.toString(), "Companion_" + c);
        }

        Seq<String> keyFields = JavaConverters.asScalaIteratorConverter(
                Arrays.asList("userid", "itemid", "rating", "Time", "Location", "Companion").iterator())
                .asScala()
                .toSeq();
        Dataset<Row> dataSetPivoted = dataSetPivotedByUser
                .join(dataSetPivotedByItem, keyFields)
                .join(dataSetPivotedByTime, keyFields)
                .join(dataSetPivotedByLocation, keyFields)
                .join(dataSetPivotedByCompanion, keyFields);

        List<String> columnsList = new ArrayList<>(Arrays.asList(dataSetPivoted.columns()));
        columnsList.removeAll(Arrays.asList(
                "userid",
                "itemid",
                "rating"));
        columnsList.removeAll(Arrays.asList(contextDimensions));

        FeaturesCount = columnsList.size();

        *//* transform columns with features to vectors *//*
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(columnsList.toArray(new String[0]))
                .setOutputCol("features");
        dataSet = assembler.transform(dataSetPivoted);*/
    }

    /**
     * rescaling each feature to range [0, 1]
     */
    private static Dataset<Row> rescaleFeatures(Dataset<Row> dataSet) {
        MinMaxScaler minMaxScaler = new MinMaxScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures");
        MinMaxScalerModel minMaxScalerModel = minMaxScaler.fit(dataSet);
        return minMaxScalerModel
                .transform(dataSet)
                .drop("features")
                .withColumnRenamed("scaledFeatures", "features");
    }

    /**
     * save data in LabeledPoint format to the LibSVM-formatted file
     */
    static void saveToLibSVM(SparkSession spark, RDD<LabeledPoint> rdd, String dataOutputFolder) throws IOException {
        /* save data in LabeledPoint format to the LibSVM-formatted file */
        FileSystem fileSystem = FileSystem.get(spark.sparkContext().hadoopConfiguration());
        Path outDir = new Path(dataOutputFolder);
        if (fileSystem.exists(outDir))
            fileSystem.delete(outDir, true);
        MLUtils.saveAsLibSVMFile(rdd.repartition(1, null), dataOutputFolder);
    }
}

