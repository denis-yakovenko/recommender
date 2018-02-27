package playground;

import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

public class LogisticRegressionWithLBFGSJava {
    public static void main(String[] args) {

        SparkSession spark = SparkSession.builder().master("local[*]").appName("CARS").getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");

        //String path = "c:/java/spark/data/mllib/sample_linear_regression_data.txt";
        //String path = "preparedData/part-00000";
        String path = "preparedDataOriginal/part-00000";
        //String path = "dataLP/part-00000";
        //String path = "part-00000-ratingnotscaled";

        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(spark.sparkContext(), path).toJavaRDD();
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3}, 11L);
        JavaRDD<LabeledPoint> training = splits[0].cache();
        JavaRDD<LabeledPoint> test = splits[1];
        LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
                .setNumClasses(3500)
                .run(training.rdd());
        JavaPairRDD<Object, Object> predictionAndLabels = test.mapToPair(p ->
                new Tuple2<>(model.predict(p.features()), p.label()));
        System.out.println("LogisticRegressionWithLBFGS predict");
        for (Tuple2<Object, Object> e : predictionAndLabels.collect())
            System.out.println(e._1 + " " + e._2);
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
        System.out.println("accuracy = " + metrics.accuracy());
        System.out.println("recall = " + metrics.recall());
        System.out.println("fMeasure = " + metrics.fMeasure());
        System.out.println("precision = " + metrics.precision());

        Dataset<Row> dataD = spark.read().format("libsvm").load(path);
        Dataset<Row>[] splitsD = dataD.randomSplit(new double[]{0.8, 0.2}, 11L);
        Dataset<Row> trainingD = splitsD[0];
        Dataset<Row> testD = splitsD[1];
        org.apache.spark.ml.classification.LogisticRegressionModel model1 = new LogisticRegression().fit(trainingD);
        JavaPairRDD<Object, Object> predictionAndLabelsD = testD.toJavaRDD().mapToPair(p -> {
                    Double predict = model1.predict(p.getAs(1));
                    Double label = p.getAs(0);
                    System.out.println(predict + " => " + label);
                    return new Tuple2<>(predict, p.getAs(0));
                }
        );
        System.out.println("LogisticRegression predict");
        for (Tuple2<Object, Object> e : predictionAndLabelsD.collect())
            System.out.println(e._1 + " " + e._2);

        MulticlassMetrics metricsD = new MulticlassMetrics(predictionAndLabelsD.rdd());
        System.out.println("accuracy = " + metricsD.accuracy());
        System.out.println("recall = " + metricsD.recall());
        System.out.println("fMeasure = " + metricsD.fMeasure());
        System.out.println("precision = " + metricsD.precision());


// Save and load model
        /*model.save(spark.sparkContext(), "target/tmp/javaLogisticRegressionWithLBFGSModel");
        LogisticRegressionModel sameModel = LogisticRegressionModel.load(spark.sparkContext(),
                "target/tmp/javaLogisticRegressionWithLBFGSModel");*/
    }
}
