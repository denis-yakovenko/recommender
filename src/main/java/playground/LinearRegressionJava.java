package playground;

import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.ml.regression.*;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class LinearRegressionJava {
    public static void main(String[] args) {


        SparkSession spark = SparkSession.builder().master("local[*]").appName("CARS").getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");

        Dataset<Row> data = spark.read().format("libsvm")
                .load(
                        "preparedDataOriginal/part-00000"
                        /*"c:/java/spark/data/mllib/"+
                        "sample_multiclass_classification_data.txt"*/
                        //"sample_linear_regression_data.txt"
                );

// Prepare training and test data.
        Dataset<Row>[] splits = data.randomSplit(new double[]{0.8, 0.2}, 11L);
        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];

        LinearRegression lr = new LinearRegression();

// We use a ParamGridBuilder to construct a grid of parameters to search over.
// TrainValidationSplit will try all combinations of values and determine best model using
// the evaluator.
        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(lr.regParam(), new double[]{1.0, 0.1, 0.01})
                .addGrid(lr.fitIntercept())
                .addGrid(lr.elasticNetParam(), new double[]{0.0, 0.5, 1.0})
                .build();

        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setMetricName("rmse");

// In this case the estimator is simply the linear regression.
// A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
        TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
                .setEstimator(lr)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(paramGrid)
                .setTrainRatio(0.8);  // 80% for training and the remaining 20% for validation

// Run train validation split, and choose the best set of parameters.
        TrainValidationSplitModel model = trainValidationSplit.fit(training);
        //model.bestModel().transform(test);
        Dataset<Row> predictions = model.bestModel().transform(test);

        double RMSE = evaluator.evaluate(predictions);
        System.out.println("Best model RMSE = " + RMSE);

        /*Double meanRating = training.select(avg("rating")).head().getDouble(0);
        Double baselineRMSE = Math.sqrt(test.select(avg(col("rating").$minus(meanRating).multiply(col("rating").$minus(meanRating)))).head().getDouble(0));
        Double improvement = (baselineRMSE - RMSE) / baselineRMSE * 100;
        System.out.println("meanRating " + meanRating + " baselineRMSE " + baselineRMSE + " best model RMSE " + RMSE);
        System.out.println("The best model improves the baseline by " + String.format("%1.2f", improvement) + "%.");
*/
// Make predictions on test data. model is the model with combination of parameters
// that performed best.
        predictions
                //.select("features", "label", "prediction")
                .show(false);
    }
}
