package playground

import java.io.File

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql
import org.apache.spark.sql.SparkSession

object SparkMLTest {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("SparkMLTest")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    val movieLensHomeDir = "Movie_DePaulMovie"
    val movieLensFile = "ratings.txt"

    var dataSet = spark
      .read
      .option("header", true)
      .option("inferSchema", true)
      .csv(new File(movieLensHomeDir, movieLensFile).toString)

    dataSet.cache()
    //dataSet.show(20, false)

    dataSet = dataSet
      .withColumn("userid", dataSet.col("userid").cast(sql.types.IntegerType))
      .withColumn("rating", dataSet.col("rating").cast(sql.types.IntegerType))
      .toDF()

    var indexer = new StringIndexer()
      .setInputCol("itemid")
      .setOutputCol("itemidI")
      .fit(dataSet)
    dataSet = indexer.transform(dataSet)

    indexer = new StringIndexer()
      .setInputCol("Time")
      .setOutputCol("TimeI")
      .fit(dataSet)
    dataSet = indexer.transform(dataSet)

    indexer = new StringIndexer()
      .setInputCol("Location")
      .setOutputCol("LocationI")
      .fit(dataSet)
    dataSet = indexer.transform(dataSet)

    indexer = new StringIndexer()
      .setInputCol("Companion")
      .setOutputCol("CompanionI")
      .fit(dataSet)
    dataSet = indexer.transform(dataSet)

    val assembler = new VectorAssembler()
      .setInputCols(Array("TimeI", "LocationI", "CompanionI"))
      .setOutputCol("features")

    val transformed = assembler.transform(dataSet)


    transformed
      .where("Time != 'NA' and Location != 'NA' and Companion != 'NA'")
      .show(20, false)

    val splits = transformed.randomSplit(Array[Double](0.8, 0.2))
    val training = splits(0)
    val test = splits(1)

    println(transformed.count())
    println(training.count())
    println(test.count())

    /*println(transformed.stat.corr("TimeI", "LocationI"))
    println(transformed.stat.corr("CompanionI", "TimeI"))
    println(transformed.stat.corr("LocationI", "CompanionI"))
    val Row(coeff: Matrix) = Correlation.corr(transformed, "features").head()
    println("pearson " + coeff.toString())
    val Row(coeff2: Matrix) = Correlation.corr(transformed, "features", "spearman").head()
    println("spearman " + coeff2.toString())*/


    /*val lr = new LogisticRegression()
      .setMaxIter(50)
      .setRegParam(0.1)
      .setElasticNetParam(0.3)
      .setLabelCol("rating")
      .setFeaturesCol("features")

    val lrModel = lr.fit(training)

    val predictions = lrModel.transform(test)
    predictions.show(predictions.count().toInt)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("rating")
      .setRawPredictionCol("prediction")
    val accuracy = evaluator.evaluate(predictions)
    println("Test set accuracy = " + accuracy)

    println(s"Coefficients:")
    println(s"${lrModel.coefficientMatrix}")
    println(s"Intercept:")
    println(s"${lrModel.interceptVector}")*/

    /*
        val km = new KMeans()
          .setK(2)
          .setSeed(1L)
          .setFeaturesCol("features")
          .setPredictionCol("cluster")

        val model = km.fit(training)
        val sum = model.computeCost(test)
        println("error is " + sum)
        println("cluster centers")
        model.clusterCenters.foreach(println)
        println("real clusters and predicted clusters")
        val predictions = model.summary.predictions
        predictions.show(predictions.count().toInt)
    */

    val alsSplits = transformed.randomSplit(Array[Double](0.8, 0.2))
    val alsTraining = splits(0)
    val alsTest = splits(1)

    val als = new ALS()
      //.setMaxIter(5)
      //.setRegParam(0.01)
      .setImplicitPrefs(true)
      .setUserCol("userid")
      .setItemCol("itemidI")
      .setRatingCol("rating")

    val ranks = List(1, 10)
    val lambdas = List(0.001, 0.01, 0.1)
    val maxIters = List(5)
    val implicitPrefs = List(true, false)

    var bestModel: ALSModel = null
    var bestValidationRMSE = Double.MaxValue
    var bestRank = 0
    var bestLambda = -1.0
    var bestMaxIter = -1
    var bestImplicitPref = false

    val alsEvaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")

    for (rank <- ranks; lambda <- lambdas; maxIter <- maxIters; implicitPref <- implicitPrefs) {

      als.setRank(rank)
      als.setMaxIter(maxIter)
      als.setImplicitPrefs(implicitPref)
      als.setRegParam(lambda)

      val alsModel = als.fit(alsTraining)
      //alsModel.setColdStartStrategy("drop")
      val alsPredictions = alsModel.transform(alsTest)
      val RMSE = alsEvaluator.evaluate(alsPredictions)
      println("RMSE = " + RMSE + ", rank = " + rank + ", lambda = " + lambda + ", maxIter = " + maxIter + ", implicitPref = " + implicitPref + ".")

      if (RMSE < bestValidationRMSE) {
        bestModel = alsModel
        bestValidationRMSE = RMSE
        bestRank = rank
        bestLambda = lambda
        bestMaxIter = maxIter
        bestImplicitPref = implicitPref
      }

    }

    /*bestModel.recommendForAllUsers(10).show(truncate = false)
    bestModel.recommendForAllItems(10).show(truncate = false)*/

    spark.stop()
  }

}
