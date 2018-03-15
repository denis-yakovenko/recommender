package recommender.contextualModeling;

import org.apache.spark.mllib.regression.FMModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static org.apache.spark.sql.functions.desc;
import static recommender.contextualModeling.ContextualModeling.getSpark;
import static recommender.contextualModeling.ContextualModeling.getPredictions;
import static recommender.contextualModeling.ContextualModeling.ratingColumnName;

public class Predict {
    public static void main(String[] args) throws IOException {
        SparkSession spark = getSpark();

        /* loading prepared dataSet */
        Dataset<Row> dataSet = spark.read().parquet("pivotedDataSet");

        /* creating an example of the user context */
        Map<String, String> userContext = new HashMap<>();
        userContext.put("Time", "Weekend");
        userContext.put("Location", "Home");
        userContext.put("Companion", "Partner");

        /* loading previously trained model */
        FMModel model = FMModel.load(spark.sparkContext(), "model");

        /* getting predictions for the user with given id returns an RDD of (itemId, rating) pairs */
        Dataset<Row> predictions = getPredictions(spark, dataSet, model, userContext, "1001");
        predictions.orderBy(desc(ratingColumnName)).show((int) predictions.count(), false);

        spark.close();
    }
}
