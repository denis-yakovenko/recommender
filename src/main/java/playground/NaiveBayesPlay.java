package playground;

import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class NaiveBayesPlay {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder().master("local[*]").appName("CARS").getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");
        Dataset<Row> dataFrame =
                spark.read().format("libsvm").load(
                        //"c:/java/spark/data/mllib/sample_libsvm_data.txt"
                        //"dataInLibSVM/part-00000"
                        "PlaygroundOutput/RatingOriginalNotScaledFeatureBinarizedSplitted"
                );
        Dataset<Row>[] splits = dataFrame.randomSplit(new double[]{0.6, 0.4}, 1234L);
        Dataset<Row> train = splits[0];
        Dataset<Row> test = splits[1];
        NaiveBayes nb = new NaiveBayes();
        NaiveBayesModel model = nb.fit(train);
        Dataset<Row> predictions = model.transform(test);
        predictions.show((int) predictions.count(), false);
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Test set accuracy = " + accuracy);
    }
}
