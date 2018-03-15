package recommender.contextualModeling;

import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.SparkSession;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Properties;

import static recommender.contextualModeling.ContextualModeling.getSpark;
import static recommender.contextualModeling.ContextualModeling.loadLibSVM;
import static recommender.contextualModeling.ContextualModeling.designModel;

public class DesignModel {
    public static void main(String[] args) throws IOException {
        SparkSession spark = getSpark();

        /* loading data in LabeledPoint format from the LibSVM-formatted file */
        RDD<LabeledPoint> convertedToLabeledPoint = loadLibSVM(spark, "libSVMDataSet");

        /* designing the Factorization Machine regression model
        given an RDD of (label, features) pairs using cross-validation with 2 folds */
        Properties modelParams = designModel(convertedToLabeledPoint, 2);

        modelParams.store(new FileOutputStream("model.properties"), "comment");

        spark.close();
    }
}
