package recommender

import java.io.File

import org.apache.hadoop.fs.{FileSystem, Hdfs, Path}

import scala.io.Source
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}

object MovieLensALS {

  def main(args: Array[String]) {

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    if (args.length != 2) {
      println("Usage: /path/to/spark/bin/spark-submit --driver-memory 2g --class MovieLensALS " +
        "target/scala-*/movielens-als-ssembly-*.jar movieLensHomeDir personalRatingsFile")
      sys.exit(1)
    }

    // set up environment

    //val conf = new SparkConf().setMaster("local[*]").setAppName("motels-home-recommendation")


    val conf = new SparkConf()
      .setAppName("MovieLensALS")
      .setMaster("local[*]")
    //.set("spark.executor.memory", "1g")
    val sc = new SparkContext(conf)
    // load personal ratings

    //val myRatings = loadRatings(args(1))
    //val myRatingsRDD = sc.parallelize(myRatings, 1)

    // load ratings and movie titles

    val movieLensHomeDir = "data"

    val ratings = sc.textFile(new File(movieLensHomeDir, "ratings.csv").toString).map { line =>
      val fields = line.split(",")
      // format: (timestamp % 10, Rating(userId, movieId, rating))
      (fields(3).toLong % 10, Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble))
    }

    // get personal ratings for the user
    val userId = 3
    //val myRatings = ratings.filter(_._2.user == userId).map(_._2).collect()
    val myRatings = ratings
      .filter(_._2.user == userId)
      .filter(_._2.product % 10 < 1)
      .map(_._2)
      .collect()
    val myRatingsRDD = sc.parallelize(myRatings, 1)

    /*val movies = sc.textFile(new File(movieLensHomeDir, "movies.csv").toString).map { line =>
      val fields = line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)")
      // format: (movieId, movieName)
      (fields(0).toInt, fields(1))
    }.collect().toMap*/

    val numRatings = ratings.count()
    val numUsers = ratings.map(_._2.user).distinct().count()
    val numMovies = ratings.map(_._2.product).distinct().count()

    println("Got " + numRatings + " ratings from "
      + numUsers + " users on " + numMovies + " movies.")

    // split ratings into train (60%), validation (20%), and test (20%) based on the
    // last digit of the timestamp, add myRatings to train, and cache them

    val numPartitions = 4
    val training = ratings.filter(x => x._1 < 6)
      .values
      .union(myRatingsRDD)
      .repartition(numPartitions)
      .cache()
    val validation = ratings.filter(x => x._1 >= 6 && x._1 < 8)
      .values
      .repartition(numPartitions)
      .cache()
    val test = ratings.filter(x => x._1 >= 8).values.cache()

    val numTraining = training.count()
    val numValidation = validation.count()
    val numTest = test.count()

    println("Training: " + numTraining + ", validation: " + numValidation + ", test: " + numTest)

    // train models and evaluate them on the validation set

    val ranks = List(1, 5, 10)
    val lambdas = List(0.001, 0.01, 0.1)
    val numIters = List(5)
    var bestModel: Option[MatrixFactorizationModel] = None
    var bestValidationRmse = Double.MaxValue
    var bestRank = 0
    var bestLambda = -1.0
    var bestNumIter = -1
    for (rank <- ranks; lambda <- lambdas; numIter <- numIters) {
      val model = ALS.train(training, rank, numIter, lambda)
      val validationRmse = computeRMSE(model, validation, numValidation)
      println("RMSE (validation) = " + validationRmse + " for the model trained with rank = "
        + rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".")
      if (validationRmse < bestValidationRmse) {
        bestModel = Some(model)
        bestValidationRmse = validationRmse
        bestRank = rank
        bestLambda = lambda
        bestNumIter = numIter
      }
    }

    val fs = FileSystem.get(sc.hadoopConfiguration)
    val outPutPath = "data/myCollaborativeFilter"
    if (fs.exists(new Path(outPutPath)))
      fs.delete(new Path(outPutPath), true)
    bestModel.get.save(sc, "data/myCollaborativeFilter")

    // evaluate the best model on the test set

    val testRMSE = computeRMSE(bestModel.get, test, numTest)

    println("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda
      + ", and numIter = " + bestNumIter + ", and its RMSE on the test set is " + testRMSE + ".")

    // create a naive baseline and compare it with the best model

    val meanRating = training.union(validation).map(_.rating).mean
    val baselineRMSE = math.sqrt(test.map(x => (meanRating - x.rating) * (meanRating - x.rating)).mean)
    val improvement = (baselineRMSE - testRMSE) / baselineRMSE * 100
    println("The best model improves the baseline by " + "%1.2f".format(improvement) + "%.")

    // make personalized recommendations

    /*val myRatedMovieIds = myRatings.map(_.product).toSet
    val candidates = sc.parallelize(movies.keys.filter(!myRatedMovieIds.contains(_)).toSeq)
    val recommendations = bestModel
      .get
      .predict(candidates.map((userId, _)))
      .collect()
      .sortBy(-_.rating)
      .take(5)

    var i = 1
    println("Movies recommended for userID = " + userId)
    recommendations.foreach { r =>
      println("%2d".format(i) + ": " + movies(r.product))
      i += 1
    }

    // make personalized recommendations using previously saved best model

    i = 1
    println("Movies recommended for userID = " + userId + " using previously saved best model")
    val model = MatrixFactorizationModel.load(sc, "data/myCollaborativeFilter")
    model
      .predict(candidates.map((userId, _)))
      .collect()
      .sortBy(r => (-r.rating, movies(r.product)))
      .take(5)
      .foreach { r =>
        println("%2d".format(i) + ": " + movies(r.product) + " - rating: " + r.rating)
        i += 1
      }*/

    // clean up
    sc.stop()
  }

  /** Compute RMSE (Root Mean Squared Error). */
  def computeRMSE(model: MatrixFactorizationModel, data: RDD[Rating], n: Long): Double = {
    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map(x => ((x.user, x.product), x.rating))
      .join(data.map(x => ((x.user, x.product), x.rating)))
      .values
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / n)
  }

  /** Load ratings from file. */
  /*def loadRatings(path: String): Seq[Rating] = {
    val lines = Source.fromFile(path).getLines()
    val ratings = lines.map { line =>
      val fields = line.split(",")
      Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
    }.filter(_.rating > 0.0)
    if (ratings.isEmpty) {
      sys.error("No ratings provided.")
    } else {
      ratings.toSeq
    }
  }*/
}
