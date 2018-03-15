package recommender.contextualModeling;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.mllib.regression.FMModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.SparkSession;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

import static recommender.contextualModeling.ContextualModeling.getSpark;
import static recommender.contextualModeling.ContextualModeling.loadLibSVM;
import static recommender.contextualModeling.ContextualModeling.trainModel;

public class TrainModel {
    public static void main(String[] args) throws IOException {
        SparkSession spark = getSpark();

        /* loading parameters of the model */
        Properties modelParams = new Properties();
        modelParams.load(new FileInputStream("model.properties"));

        /* loading data in LabeledPoint format from the LibSVM-formatted file */
        RDD<LabeledPoint> convertedToLabeledPoint = loadLibSVM(spark, "libSVMDataSet");

        /* training the Factorization Machine regression model given an RDD of (label, features) pairs */
        FMModel model = trainModel(convertedToLabeledPoint, modelParams);

        /* saving trained model */
        FileSystem fileSystem = FileSystem.get(spark.sparkContext().hadoopConfiguration());
        Path outDir = new Path("model");
        if (fileSystem.exists(outDir))
            fileSystem.delete(outDir, true);
        model.save(spark.sparkContext(), "model");

        spark.close();
    }
}
