package playground

import java.io.File

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.rdd._
import org.apache.spark.{SparkConf, SparkContext}

object MovieLensTransform {

  def main(args: Array[String]) {

    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    val conf = new SparkConf()
      .setAppName("MovieItemSplittingALS")
      .setMaster("local[*]")
    val sc = new SparkContext(conf)
    val movieLensHomeDir = "Movie_DePaulMovie"
    val movieLensFile = "ratings.txt"
    /*val movieLensHomeDir = "data"
    val movieLensFile = "ratings.csv"*/

    /* loading source data */
    val source = sc.textFile(new File(movieLensHomeDir, movieLensFile).toString)
      .mapPartitionsWithIndex((idx, iter) => if (idx == 0) iter.drop(1) else iter)
      .map { line =>
        val fields = line.split(",")
        (
          fields(0).toInt,
          fields(1),
          fields(2).toDouble,
          fields(3),
          fields(4),
          fields(5)
          )
      }

    /*println("source")
    source.filter(_._1 == 1).foreach(println)*/

    /* adding context dimensions to the source data using item splitting method */
    val sourceWithSplitItems = source.map { line =>
      (
        line._1,
        line._2
          .concat("\t").concat(line._4)
          .concat("\t").concat(line._5)
          .concat("\t").concat(line._6)
        ,
        line._3.toDouble
        )
    }

    /*println("sourceWithSplitItems")
    sourceWithSplitItems.filter(_._1 == 1).foreach(println)*/

    /* collecting unique items and indexing with Ids to store as Rating object */
    val itemsIndexed = sourceWithSplitItems.map(_._2).distinct().zipWithIndex().map(x => (-x._2.toInt, x._1))

    /*println("itemsIndexed")
    itemsIndexed.filter(_._2.equals("31")).foreach(println)*/

    /* creating ratings and replacing item's names with indexed Ids */
    val r = scala.util.Random
    val ratings = sourceWithSplitItems.keyBy(_._2).join(itemsIndexed.keyBy(_._2))
      .map(
        x => ((r.nextFloat() * 10.0).toInt % 10, Rating(
          x._2._1._1,
          x._2._2._1,
          x._2._1._3
        ))
      )

    /*println("ratings")
    ratings.filter(_._2.user == 1).foreach(println)*/

    /* creating the list of movies*/
    val movies = itemsIndexed.map(x => (x._1, x._2)).collect().toMap

    /*println("movies")
    movies.foreach(println)*/

    /* getting personal ratings for the random user */
    val userId = ratings.takeSample(withReplacement = false, 1)(0)._2.user

    /* creating list of the movies that the user has already seen */
    val myRatings = ratings
      .filter(_._2.user == userId)
      .filter(_._2.product % 10 < 1)
      .map(_._2)
      .collect()
    val myRatingsRDD = sc.parallelize(myRatings, 1)

    /*println("myRatings for userId: " + userId)
    myRatings.foreach(println)*/

    val numRatings = ratings.count()
    val numUsers = ratings.map(_._2.user).distinct().count()
    val numMovies = ratings.map(_._2.product).distinct().count()

    println("Got " + numRatings + " ratings from " + numUsers + " users on " + numMovies + " movies.")

    /* split ratings into train (60%), validation (20%), and test (20%), add myRatings to train, and cache them */
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

    /* train models and evaluate them on the validation set */
    val ranks = List(1, 5, 10)
    val lambdas = List(0.001, 0.01, 0.1)
    val numIters = List(5, 10)
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

    /* save the best model */
    val fs = FileSystem.get(sc.hadoopConfiguration)
    val outPutPath = "data/myCollaborativeFilter"
    if (fs.exists(new Path(outPutPath)))
      fs.delete(new Path(outPutPath), true)
    bestModel.get.save(sc, "data/myCollaborativeFilter")

    /* evaluate the best model on the test set */
    val testRMSE = computeRMSE(bestModel.get, test, numTest)
    println("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda
      + ", and numIter = " + bestNumIter + ", and its RMSE on the test set is " + testRMSE + ".")

    /* create a naive baseline and compare it with the best model */
    val meanRating = training.union(validation).map(_.rating).mean
    val baselineRMSE = math.sqrt(test.map(x => (meanRating - x.rating) * (meanRating - x.rating)).mean)
    val improvement = (baselineRMSE - testRMSE) / baselineRMSE * 100
    println("The best model improves the baseline by " + "%1.2f".format(improvement) + "%.")

    /* make personalized recommendations */
    val myRatedMovieIds = myRatings.map(_.product).toSet
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
      println("%2d".format(i) + ": " + movies(r.product) + " - rating: " + r.rating)
      i += 1
    }

    /* make personalized recommendations using previously saved best model */
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
      }

    /* clean up */
    sc.stop()

  }

  def computeRMSE(model: MatrixFactorizationModel, data: RDD[Rating], n: Long): Double = {
    val predictions: RDD[Rating] = model
      .predict(
        data.map(x => (x.user, x.product))
      )
    val predictionsAndRatings = predictions
      .map(
        x => ((x.user, x.product), x.rating)
      )
      .join(data.map(x => ((x.user, x.product), x.rating)))
      .values
    math
      .sqrt(
        predictionsAndRatings
          .map(
            x => (x._1 - x._2) * (x._1 - x._2)
          )
          .reduce(_ + _) / n
      )
  }
}
