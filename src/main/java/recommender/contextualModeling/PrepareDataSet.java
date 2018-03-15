package recommender.contextualModeling;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;

import static recommender.contextualModeling.ContextualModeling.getSpark;
import static recommender.contextualModeling.ContextualModeling.saveToLibSVM;
import static recommender.contextualModeling.ContextualModeling.*;

public class PrepareDataSet {

    public static void main(String[] args) throws IOException {
        SparkSession spark = getSpark();

        /* loading data set */
        Dataset<Row> dataSet = loadDataSet(spark, "Movie_DePaulMovie/ratingsOriginal.txt");

        /* showing dataSet total counts */
        showStatistics(dataSet);

        /* rescaling rating to range [-1, 1] */
        dataSet = rescaleRatings(dataSet);

        /* pivoting values to columns and transform to vectors of features */
        dataSet = pivotRowsToFeatures(dataSet);

        /* converting data to LabeledPoint to train model */
        RDD<LabeledPoint> convertedToLabeledPoint = convertToLabeledPoint(dataSet);

        FileSystem fileSystem = FileSystem.get(spark.sparkContext().hadoopConfiguration());

        String libSVMDataSetFolder = "libSVMDataSet";
        Path outDir = new Path(libSVMDataSetFolder);
        if (fileSystem.exists(outDir)) fileSystem.delete(outDir, true);
        saveToLibSVM(spark, convertedToLabeledPoint, libSVMDataSetFolder);

        String pivotedDataSetFolder = "pivotedDataSet";
        outDir = new Path(pivotedDataSetFolder);
        if (fileSystem.exists(outDir)) fileSystem.delete(outDir, true);
        dataSet.write().parquet(pivotedDataSetFolder);

        spark.close();
    }
}
