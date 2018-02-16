package recommender;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.File;
import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.col;

public class SparkMLTestJava2 {

    public static class Rating implements Serializable {
        private int userId;
        private int movieId;
        private float rating;
        private long timestamp;

        public Rating() {
        }

        public Rating(int userId, int movieId, float rating, long timestamp) {
            this.userId = userId;
            this.movieId = movieId;
            this.rating = rating;
            this.timestamp = timestamp;
        }

        public int getUserId() {
            return userId;
        }

        public int getMovieId() {
            return movieId;
        }

        public float getRating() {
            return rating;
        }

        public long getTimestamp() {
            return timestamp;
        }

        public static Rating parseRating(String str) {
            String[] fields = str.split(",");
            if (fields.length != 4) {
                throw new IllegalArgumentException("Each line must contain 4 fields");
            }
            int userId = Integer.parseInt(fields[0]);
            int movieId = Integer.parseInt(fields[1]);
            float rating = Float.parseFloat(fields[2]);
            long timestamp = Long.parseLong(fields[3]);
            return new Rating(userId, movieId, rating, timestamp);
        }
    }

    public static void main(String[] args) {

        SparkSession spark = SparkSession
                .builder()
                .master("local[*]")
                .appName("CARS")
                .getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");

        //Dataset<Row> dataSet = getDePaulMovieDataSet(spark);
        Dataset<Row> dataSet = getML100KDataSet(spark);

        dataSet.printSchema();
        dataSet.show(20, false);
/*
        spark.stop();
        System.exit(0);
*/

/*
        StringIndexerModel itemIndexer = new StringIndexer()
                .setInputCol("itemid")
                .setOutputCol("itemidIndexed")
                .fit(dataSet);
        dataSet = itemIndexer.transform(dataSet);

        StringIndexerModel userIndexer = new StringIndexer()
                .setInputCol("userid")
                .setOutputCol("useridIndexed")
                .fit(dataSet);
        dataSet = userIndexer.transform(dataSet);

        dataSet.cache();
        dataSet.show(false);
        dataSet.printSchema();
*/

        /** contextual pre-filtering */
        /*Map<String, String> userContext = new HashMap<>();
        userContext.put("Time", "Weekend");
        userContext.put("Location", "Home");
        userContext.put("Companion", "Family");
        userContext.clear();
        for (Map.Entry<String, String> contextVar : userContext.entrySet()) {
            dataSet = dataSet.filter(col(contextVar.getKey()).equalTo(contextVar.getValue()));
        }*/

        Long numRatings = dataSet.count();
        Long numUsers = dataSet.select("userid").distinct().count();
        Long numMovies = dataSet.select("itemid").distinct().count();
        System.out.println("Got " + numRatings + " ratings from " + numUsers + " users on " + numMovies + " movies.");

        Dataset<Row>[] splits = dataSet.randomSplit(new double[]{0.6, 0.2, 0.2});
        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];
        Dataset<Row> validation = splits[2];
        Long numTraining = training.count();
        Long numTest = test.count();
        Long numValidation = validation.count();
        System.out.println("Training: " + numTraining + ", test: " + numTest + ", validation: " + numValidation);

        /** train models and evaluate them on the validation set */
        ALS als = new ALS()
                .setMaxIter(5)
                .setRegParam(0.01)
                //.setAlpha()
                .setRank(1)
                //.setSeed()
                //.setImplicitPrefs(true)
/*
                .setUserCol("useridIndexed")
                .setItemCol("itemidIndexed")
*/
                .setUserCol("userid")
                .setItemCol("itemid")
                .setRatingCol("rating");
        RegressionEvaluator alsEvaluator = new RegressionEvaluator()
                .setMetricName("rmse")
                .setLabelCol("rating")
                .setPredictionCol("prediction");
        ALSModel model = als.fit(training);
        model.setColdStartStrategy("drop");
        Dataset<Row> alsPredictions = model.transform(test);
        double RMSE = alsEvaluator.evaluate(alsPredictions);
        System.out.println("RMSE = " + RMSE);

        /** create a naive baseline and compare it with the model */
        Double meanRating =
                training.union(validation).select(avg("rating")).head().getDouble(0);
        Double baselineRMSE = Math.sqrt(
                test.select(avg(col("rating").$minus(meanRating).multiply(col("rating").$minus(meanRating)))).head().getDouble(0)
        );
        Double improvement = (baselineRMSE - RMSE) / baselineRMSE * 100;
        System.out.println("meanRating " + meanRating + " baselineRMSE " + baselineRMSE + " best model RMSE " + RMSE);
        System.out.println("The model improves the baseline by " + String.format("%1.2f", improvement) + "%.");

        /*IndexToString itemBackIndexer = new IndexToString()
                .setInputCol("itemid")
                .setOutputCol("itemidIndexed")
                .setLabels(itemIndexer.labels());
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{itemBackIndexer});
        PipelineModel model = pipeline.fit(training);
        Dataset<Row> predictions = model.transform(test);
        predictions.select("itemidIndexed", "itemid").show(5);*/

        /** get recommendations for all users */
        Dataset<Row> recommendForAllUsers = model.recommendForAllUsers(5);
        Dataset<Row> recommendForAllItems = model.recommendForAllItems(5);
        recommendForAllUsers.show(false);
        recommendForAllItems.show(false);

        /** contextual post-filtering */
        //recommendForAllUsers.head().get()
        /*Map<String, String> userContext = new HashMap<>();
        userContext.put("Time", "Weekend");
        userContext.put("Location", "Home");
        userContext.put("Companion", "Family");
        for (Map.Entry<String, String> contextVar : userContext.entrySet()) {
            dataSet = dataSet.filter(col(contextVar.getKey()).equalTo(contextVar.getValue()));
        }*/

        spark.stop();
    }

    static Dataset<Row> getDePaulMovieDataSet(SparkSession spark) {
        return spark
                .read()
                .option("header", true)
                .option("inferSchema", true)
                .csv(new File("Movie_DePaulMovie", "ratings.txt").toString());
    }

    static Dataset<Row> getML100KDataSet(SparkSession spark) {


        JavaRDD<Rating> ratingsRDD = spark
                .read().textFile("data/ratings_small.csv").javaRDD()
                .map(Rating::parseRating);
        Dataset<Row> ratings = spark.createDataFrame(ratingsRDD, Rating.class);

        return ratings
                .withColumnRenamed("userId", "userid")
                .withColumnRenamed("movieId", "itemid")
                .toDF();

/*
        Dataset<Row> dataSet = spark
                .read()
                .csv(new File("data", "ratings_small.csv").toString());
                //.toJavaRDD().map(line->);
        dataSet.registerTempTable("dataSet");
*/
        /*dataSet = dataSet
                .withColumn("userid", dataSet.col("userid").cast(IntegerType))
                .withColumn("rating", dataSet.col("rating").cast(IntegerType))
                .toDF();*/
        //return spark.sql("select cast(_c0 as int) as userid, cast(_c1 as int) as itemid, cast(_c2 as float) as rating from dataset");
    }
}

