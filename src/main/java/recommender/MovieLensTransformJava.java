package recommender;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import scala.Tuple2;
import scala.Tuple3;
import scala.Tuple6;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

public class MovieLensTransformJava {

    public static void main(String[] args) throws IOException {

        Logger.getLogger("org.apache.spark").setLevel(Level.WARN);
        Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF);
        SparkConf conf = new SparkConf()
                .setAppName("MovieItemSplittingALS")
                .setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);
        String movieLensHomeDir = "Movie_DePaulMovie";
        String movieLensFile = "ratings.txt";
        /*val movieLensHomeDir = "data"
        val movieLensFile = "ratings.csv"*/

        /** loading source data */
        JavaRDD<Tuple6<Integer, String, Double, String, String, String>> source = sc
                .textFile(movieLensHomeDir + "/" + movieLensFile)
                .mapPartitionsWithIndex((Function2<Integer, Iterator<String>, Iterator<String>>) (index, iterator) -> {
                    if (index == 0 && iterator.hasNext())
                        iterator.next();
                    return iterator;
                }, false)
                .map((Function<String, Tuple6<Integer, String, Double, String, String, String>>) s -> {
                    String[] fields = s.split(",");
                    return new Tuple6<>(
                            Integer.valueOf(fields[0]),
                            fields[1],
                            Double.valueOf(fields[2]),
                            fields[3],
                            fields[4],
                            fields[5]);
                });
        System.out.println("source");
        source.take(3).forEach(System.out::println);

        /** adding context dimensions to the source data using item splitting method */
        JavaRDD<Tuple3<Integer, String, Double>> sourceWithSplitItems = source.map(line -> new Tuple3<>(
                line._1(),
                line._2()
                        .concat("\t").concat(line._4())
                        .concat("\t").concat(line._5())
                        .concat("\t").concat(line._6())
                ,
                line._3()));
        System.out.println("sourceWithSplitItems");
        sourceWithSplitItems.take(3).forEach(System.out::println);

        /** collecting unique items and indexing with Ids to store as Rating object */
        JavaRDD<Tuple2<Integer, String>> itemsIndexed = sourceWithSplitItems
                .map(Tuple3::_2)
                .distinct()
                .zipWithIndex()
                .map(x -> new Tuple2<>(-x._2.intValue(), x._1));
        System.out.println("itemsIndexed");
        itemsIndexed.take(3).forEach(System.out::println);

        /** creating ratings and replacing item's names with indexed Ids */
        Random r = new Random();
        JavaRDD<Tuple2<Integer, Rating>> ratings = sourceWithSplitItems.keyBy(Tuple3::_2).join(itemsIndexed.keyBy(Tuple2::_2))
                .map(
                        x -> new Tuple2<>(
                                (int) ((r.nextFloat() * 10.0) % 10),
                                new Rating(
                                        x._2._1._1(),
                                        x._2._2._1(),
                                        x._2._1._3()
                                ))
                );
        System.out.println("ratings");
        ratings.take(3).forEach(System.out::println);

        /** creating the list of movies*/
        Map<Integer, String> movies = itemsIndexed
                .map(x -> new Tuple2<>(x._1, x._2))
                .collect()
                .stream()
                .collect(Collectors.toMap(Tuple2::_1, Tuple2::_2));
        System.out.println("movies");
        movies.entrySet().stream().limit(3).forEach(v -> System.out.println("key: " + v.getKey() + ", value: " + v.getValue()));

        /** getting personal ratings for the random user */
        Integer userId = ratings.takeSample(false, 1).get(0)._2.user();

        /** creating list of the movies that the user has already seen */
        JavaRDD<Rating> myRatingsRDD = ratings
                .filter(x -> x._2.user() == userId)
                .filter(x -> x._2.product() % 10 < 1)
                .map(x -> x._2);
        System.out.println("myRatings for userId: " + userId);
        myRatingsRDD.take(3).forEach(System.out::println);
        Long numRatings = ratings.count();
        Long numUsers = ratings.map(x -> x._2.user()).distinct().count();
        Long numMovies = ratings.map(x -> x._2.product()).distinct().count();
        System.out.println("Got " + numRatings + " ratings from " + numUsers + " users on " + numMovies + " movies.");

        /** split ratings into train (60%), validation (20%), and test (20%), add myRatings to train, and cache them */
        Integer numPartitions = 4;
        JavaRDD<Rating> training = ratings.filter(x -> x._1 < 6)
                .map(x -> x._2)
                .union(myRatingsRDD)
                .repartition(numPartitions)
                .cache();
        JavaRDD<Rating> validation = ratings.filter(x -> x._1 >= 6 && x._1 < 8)
                .map(x -> x._2)
                .repartition(numPartitions)
                .cache();
        JavaRDD<Rating> test = ratings.filter(x -> x._1 >= 8)
                .map(x -> x._2)
                .cache();
        Long numTraining = training.count();
        Long numValidation = validation.count();
        Long numTest = test.count();
        System.out.println("Training: " + numTraining + ", validation: " + numValidation + ", test: " + numTest);

        /** train models and evaluate them on the validation set */
        List<Integer> ranks = Arrays.asList(1, 5, 10);
        List<Double> lambdas = Arrays.asList(1.0, 0.1);
        List<Integer> numIters = Arrays.asList(/*5, */10);
        MatrixFactorizationModel bestModel = null;
        Double bestValidationRmse = Double.MAX_VALUE;
        Integer bestRank = 0;
        Double bestLambda = -1.0;
        Integer bestNumIter = -1;
        for (Integer rank : ranks) {
            for (Double lambda : lambdas) {
                for (Integer numIter : numIters) {
                    MatrixFactorizationModel model = ALS.train(training.rdd(), rank, numIter, lambda);
                    Double validationRmse = computeRMSE(model, validation, numValidation);
                    System.out.println("RMSE (validation) = " + validationRmse + " for the model trained with rank = "
                            + rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".");
                    if (validationRmse < bestValidationRmse) {
                        bestModel = model;
                        bestValidationRmse = validationRmse;
                        bestRank = rank;
                        bestLambda = lambda;
                        bestNumIter = numIter;
                    }
                }
            }
        }

        /** save the best model */
        FileSystem fs = FileSystem.get(sc.hadoopConfiguration());
        String outPutPath = "data/myCollaborativeFilter";
        if (fs.exists(new Path(outPutPath)))
            fs.delete(new Path(outPutPath), true);
        if (bestModel != null) {
            bestModel.save(sc.sc(), "data/myCollaborativeFilter");
        }

        /** evaluate the best model on the test set */
        Double testRMSE = computeRMSE(bestModel, test, numTest);
        System.out.println("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda
                + ", and numIter = " + bestNumIter + ", and its RMSE on the test set is " + testRMSE + ".");

        /** create a naive baseline and compare it with the best model */
        Double meanRating = training.union(validation).mapToDouble(Rating::rating).mean();
        Double baselineRMSE = Math.sqrt(test.mapToDouble(x -> (meanRating - x.rating()) * (meanRating - x.rating())).mean());
        Double improvement = (baselineRMSE - testRMSE) / baselineRMSE * 100;
        System.out.println("meanRating " + meanRating + " baselineRMSE " + baselineRMSE + " best model RMSE " + testRMSE);
        System.out.println("The best model improves the baseline by " + String.format("%1.2f", improvement) + "%.");

        /** make personalized recommendations */
        Set<Integer> myRatedMovieIds = new HashSet<>(myRatingsRDD.map(Rating::product).collect());
        List<Integer> integerList = movies.keySet().stream().filter(x -> !myRatedMovieIds.contains(x)).collect(Collectors.toList());
        JavaRDD<Integer> candidates = sc.parallelize(integerList);
        assert bestModel != null;
        List<Rating> recommendations =
                new ArrayList<>(bestModel
                        .predict(JavaRDD.toRDD(candidates.map(x -> new Tuple2<>(userId, x))))
                        .toJavaRDD()
                        .collect());
        recommendations.sort((o1, o2) -> Double.compare(o2.rating(), o1.rating()));
        AtomicInteger i = new AtomicInteger(1);
        System.out.println("Movies recommended for userID = " + userId);
        recommendations.stream().limit(5).forEach(
                x -> System.out.println(String.format("%2d", i.getAndIncrement()) + ": " + movies.get(x.product()) + " - rating: " + x.rating())
        );

        /** make personalized recommendations using previously saved best model */
        i.set(1);
        System.out.println("Movies recommended for userID = " + userId + " using previously saved best model");
        MatrixFactorizationModel model = MatrixFactorizationModel.load(sc.sc(), "data/myCollaborativeFilter");
        recommendations = new ArrayList<>(model
                .predict(JavaRDD.toRDD(candidates.map(x -> new Tuple2<>(userId, x))))
                .toJavaRDD()
                .collect());
        recommendations.sort(
                Comparator
                        .comparing(Rating::rating)
                        .reversed()
                        .thenComparing(
                                Rating::product, (s1, s2) -> movies.get(s1).compareTo(movies.get(s2))
                        )
        );
        recommendations.stream().limit(5).forEach(
                x -> System.out.println(String.format("%2d", i.getAndIncrement()) + ": " + movies.get(x.product()) + " - rating: " + x.rating())
        );

        /** clean up */
        sc.stop();

    }

    private static Double computeRMSE(MatrixFactorizationModel model, JavaRDD<Rating> data, Long n) {
        JavaRDD<Rating> predictions =
                model
                        .predict(JavaRDD.toRDD(data.map(
                                x -> new Tuple2<>(x.user(), x.product())
                                ))
                        ).toJavaRDD();
        JavaRDD<Tuple2<Double, Double>> predictionsAndRatings =
                JavaPairRDD.fromJavaRDD(predictions.map(
                        r -> new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating())
                ))
                        .join(JavaPairRDD.fromJavaRDD(

                                data.map(r -> new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating()))

                        ))
                        .values();
        return Math
                .sqrt(
                        predictionsAndRatings
                                .map(
                                        x -> (x._1 - x._2) * (x._1 - x._2)
                                )
                                .reduce((a, b) -> a + b) / n
                );
    }
}
