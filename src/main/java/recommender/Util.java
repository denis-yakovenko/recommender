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

import java.io.File;
import java.io.IOException;

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
     * save data in LabeledPoint format to the LibSVM-formatted file
     */
    static void saveToLibSVM(SparkSession spark, RDD<LabeledPoint> dataLP, String dataOutputFolder) throws IOException {
        /* save data in LabeledPoint format to the LibSVM-formatted file */
        FileSystem fileSystem = FileSystem.get(spark.sparkContext().hadoopConfiguration());
        Path outDir = new Path(dataOutputFolder);
        if (fileSystem.exists(outDir))
            fileSystem.delete(outDir, true);
        MLUtils.saveAsLibSVMFile(dataLP.repartition(1, null), dataOutputFolder);
    }

    /**
     * saving/loading libsvm
     */
    static RDD<LabeledPoint> loadLibSVM(SparkSession spark) {
        return MLUtils
                .loadLibSVMFile(
                        spark.sparkContext(),
                        "PlaygroundOutput/RatingOriginalNotScaledFeatureBinarizedSplitted"
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
}