package recommender;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.ml.feature.*;
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
import java.util.Arrays;

import static org.apache.spark.sql.functions.*;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.lit;
import static org.apache.spark.sql.types.DataTypes.DoubleType;
import static org.apache.spark.sql.types.DataTypes.IntegerType;

public class Util {

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
     * rescaling each feature to range [0, 1]
     */
    static Dataset<Row> getWithRescaledFeatures(Dataset<Row> dataSet) {
        /* rescaling each feature to range [0, 1] */
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
     * showing dataSet total counts
     */
    static void showDataSetStatistics(Dataset<Row> dataSet) {
        /* showing dataSet total counts */
        Long numRatings = dataSet.count();
        Long numUsers = dataSet.select("userid").distinct().count();
        Long numMovies = dataSet.select("itemid").distinct().count();
        System.out.println("Got " + numRatings + " ratings from " + numUsers + " users on " + numMovies + " movies.");
    }

    static Dataset<Row> getPivotedToColumns(Dataset<Row> dataSet) {
        /* pivoting values to columns */
        /*Dataset<Row> dataSetTime = dataSet
                .groupBy(col("useridI"), col("itemidI"), col("rating"))
                .pivot("Time", dataSet.select("Time").distinct().toJavaRDD().map(r -> r.getAs("Time")).collect()) /////////////
                .agg(lit(1));*/

        /* pivoting user values to columns */
        Dataset<Row> dataSetPivotedByUser = dataSet
                .groupBy(col("userid"), col("itemid"), col("rating"))
                .pivot("userid")
                .agg(lit(1));
        dataSetPivotedByUser = dataSetPivotedByUser.na().fill(0.0);
        for (Object c : dataSet.select("userid").distinct().toJavaRDD().map(r -> r.getAs("userid")).collect()) {
            dataSetPivotedByUser = dataSetPivotedByUser.withColumnRenamed(c.toString(), "user_" + c);
            dataSetPivotedByUser = dataSetPivotedByUser.withColumn("user_" + c, col("user_" + c).cast("double"));
        }
        /*dataSetPivotedByUser.show(false);
        dataSetPivotedByUser.printSchema();*/

        /* pivoting item values to columns */
        Dataset<Row> dataSetPivotedByItem = dataSet
                .groupBy(col("userid"), col("itemid"), col("rating"))
                .pivot("itemid")
                .agg(lit(1));
        dataSetPivotedByItem = dataSetPivotedByItem.na().fill(0.0);
        for (Object c : dataSet.select("itemid").distinct().toJavaRDD().map(r -> r.getAs("itemid")).collect()) {
            dataSetPivotedByItem = dataSetPivotedByItem.withColumnRenamed(c.toString(), "item_" + c);
        }
        /*dataSetPivotedByItem.show(false);
        dataSetPivotedByItem.printSchema();*/

        /* pivoting Time values to columns */
        Dataset<Row> dataSetPivotedByTime = dataSet
                .groupBy(col("userid"), col("itemid"), col("rating"))
                .pivot("Time")
                .agg(lit(1));
        dataSetPivotedByTime = dataSetPivotedByTime.na().fill(0.0);
        for (Object c : dataSet.select("Time").distinct().toJavaRDD().map(r -> r.getAs("Time")).collect()) {
            dataSetPivotedByTime = dataSetPivotedByTime.withColumnRenamed(c.toString(), "Time_" + c);
        }
        /*dataSetPivotedByTime.show(false);
        dataSetPivotedByTime.printSchema();*/

        /* pivoting Location values to columns */
        Dataset<Row> dataSetPivotedByLocation = dataSet
                .groupBy(col("userid"), col("itemid"), col("rating"))
                .pivot("Location")
                .agg(lit(1));
        dataSetPivotedByLocation = dataSetPivotedByLocation.na().fill(0.0);
        for (Object c : dataSet.select("Location").distinct().toJavaRDD().map(r -> r.getAs("Location")).collect()) {
            dataSetPivotedByLocation = dataSetPivotedByLocation.withColumnRenamed(c.toString(), "Location_" + c);
        }
        /*dataSetPivotedByLocation.show(false);
        dataSetPivotedByLocation.printSchema();*/

        /* pivoting Companion values to columns */
        Dataset<Row> dataSetPivotedByCompanion = dataSet
                .groupBy(col("userid"), col("itemid"), col("rating"))
                .pivot("Companion")
                .agg(lit(1));
        dataSetPivotedByCompanion = dataSetPivotedByCompanion.na().fill(0.0);
        for (Object c : dataSet.select("Companion").distinct().toJavaRDD().map(r -> r.getAs("Companion")).collect()) {
            dataSetPivotedByCompanion = dataSetPivotedByCompanion.withColumnRenamed(c.toString(), "Companion_" + c);
        }
        /*dataSetPivotedByCompanion.show(false);
        dataSetPivotedByCompanion.printSchema();*/

        /* joining pivoted data sets into one */
        Seq<String> keyFields = JavaConverters.asScalaIteratorConverter(Arrays.asList("userid", "itemid", "rating").iterator()).asScala().toSeq();
        Dataset<Row> dataSetPivoted = dataSetPivotedByUser
                .join(dataSetPivotedByItem, keyFields)
                .join(dataSetPivotedByTime, keyFields)
                .join(dataSetPivotedByLocation, keyFields)
                .join(dataSetPivotedByCompanion, keyFields);

        /*dataSetPivoted.show(false);
        dataSetPivoted.printSchema();*/

        /* transform columns with features to vectors */
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(
                        ArrayUtils.removeElement(
                                ArrayUtils.removeElement(
                                        ArrayUtils.removeElement(dataSetPivoted.columns(),
                                                "userid"),
                                        "itemid"),
                                "rating")
                )
                .setOutputCol("features");
        dataSetPivoted = assembler.transform(dataSetPivoted);
        return dataSetPivoted;
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
     * rescaling rating to range [0, 1]
     */
    static Dataset<Row> getWithRescaledRating(Dataset<Row> dataSet) {
        /* rescaling rating to range [0, 1] */
        Row rowMinMax = dataSet.agg(
                min(col("rating")),
                max(col("rating"))).head();
        Double minRating = rowMinMax.getAs(0);
        Double maxRating = rowMinMax.getAs(1);
        return dataSet
                .withColumn(
                        "rating",
                        (col("rating").minus(minRating)).divide(maxRating - minRating)
                );
    }
}