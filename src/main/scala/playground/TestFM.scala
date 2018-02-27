package playground

import org.apache.spark.mllib.regression.FMWithLBFGS
import org.apache.spark.sql.SparkSession

object TestFM {

  def main(args: Array[String]): Unit = {

    val sc = SparkSession.builder.master("local[*]").appName("CARS").getOrCreate
    sc.sparkContext.setLogLevel("ERROR")
    //var training = MLUtils.loadLibSVMFile(sc.sparkContext, "housing_scale.txt").cache()
/*
    val data = MLUtils.loadLibSVMFile(
      sc.sparkContext,
      //RatingOriginalNotScaledFeatureBinarizedSplitted
      //RatingOriginalScaledFeatureBinarizedSplitted
      //RatingWeightedNotScaledFeatureBinarizedSplitted
      //RatingWeightedNotScaledFeatureNotScaled
      //RatingWeightedNotScaledFeatureScaled
      //RatingWeightedScaledFeatureBinarizedSplitted
      //RatingWeightedScaledFeatureScaled
      //RatingOriginalScaledFeatureNotScaled
      //RatingOriginalScaledFeatureScaled
      "out/RatingOriginalScaledFeatureBinarizedSplitted"
      //"ratings_binary.txt"
    ).cache()
*/




    val data = sc.sparkContext.parallelize(List(
      //                                   u1 u2 u3 i1 i2 c1
      new org.apache.spark.mllib.regression.LabeledPoint(0.90, org.apache.spark.mllib.linalg.Vectors.dense(0.1, 0.0, 0.0, 1.0, 0.0, 1.0)),
      new org.apache.spark.mllib.regression.LabeledPoint(0.80, org.apache.spark.mllib.linalg.Vectors.dense(0.1, 0.0, 0.0, 1.0, 0.0, 0.0)),
      new org.apache.spark.mllib.regression.LabeledPoint(0.60, org.apache.spark.mllib.linalg.Vectors.dense(0.1, 0.0, 0.0, 0.0, 1.0, 1.0)),
      new org.apache.spark.mllib.regression.LabeledPoint(0.65, org.apache.spark.mllib.linalg.Vectors.dense(0.1, 0.0, 0.0, 0.0, 1.0, 0.0)),
      new org.apache.spark.mllib.regression.LabeledPoint(0.85, org.apache.spark.mllib.linalg.Vectors.dense(0.0, 0.3, 0.0, 1.0, 0.0, 1.0)),
      new org.apache.spark.mllib.regression.LabeledPoint(0.80, org.apache.spark.mllib.linalg.Vectors.dense(0.0, 0.3, 0.0, 1.0, 0.0, 0.0)),
      new org.apache.spark.mllib.regression.LabeledPoint(0.55, org.apache.spark.mllib.linalg.Vectors.dense(0.0, 0.3, 0.0, 0.0, 1.0, 1.0)),
      new org.apache.spark.mllib.regression.LabeledPoint(0.50, org.apache.spark.mllib.linalg.Vectors.dense(0.0, 0.3, 0.0, 0.0, 1.0, 0.0)),
      new org.apache.spark.mllib.regression.LabeledPoint(0.77, org.apache.spark.mllib.linalg.Vectors.dense(0.0, 0.0, 0.9, 1.0, 0.0, 1.0)),
      new org.apache.spark.mllib.regression.LabeledPoint(0.77, org.apache.spark.mllib.linalg.Vectors.dense(0.0, 0.0, 0.9, 1.0, 0.0, 0.0)),
      new org.apache.spark.mllib.regression.LabeledPoint(1.43, org.apache.spark.mllib.linalg.Vectors.dense(0.0, 0.0, 0.9, 0.0, 1.0, 1.0)),
      new org.apache.spark.mllib.regression.LabeledPoint(1.48, org.apache.spark.mllib.linalg.Vectors.dense(0.0, 0.0, 0.9, 0.0, 1.0, 0.0))
/*
      new org.apache.spark.mllib.regression.LabeledPoint(0.90, org.apache.spark.mllib.linalg.Vectors.dense(1, 0, 0, 1, 0, 1)),
      new org.apache.spark.mllib.regression.LabeledPoint(0.80, org.apache.spark.mllib.linalg.Vectors.dense(1, 0, 0, 1, 0, 0)),
      new org.apache.spark.mllib.regression.LabeledPoint(0.60, org.apache.spark.mllib.linalg.Vectors.dense(1, 0, 0, 0, 1, 1)),
      new org.apache.spark.mllib.regression.LabeledPoint(0.65, org.apache.spark.mllib.linalg.Vectors.dense(1, 0, 0, 0, 1, 0)),
      new org.apache.spark.mllib.regression.LabeledPoint(0.85, org.apache.spark.mllib.linalg.Vectors.dense(0, 1, 0, 1, 0, 1)),
      new org.apache.spark.mllib.regression.LabeledPoint(0.80, org.apache.spark.mllib.linalg.Vectors.dense(0, 1, 0, 1, 0, 0)),
      new org.apache.spark.mllib.regression.LabeledPoint(0.55, org.apache.spark.mllib.linalg.Vectors.dense(0, 1, 0, 0, 1, 1)),
      new org.apache.spark.mllib.regression.LabeledPoint(0.50, org.apache.spark.mllib.linalg.Vectors.dense(0, 1, 0, 0, 1, 0)),
      new org.apache.spark.mllib.regression.LabeledPoint(0.77, org.apache.spark.mllib.linalg.Vectors.dense(0, 0, 1, 1, 0, 1)),
      new org.apache.spark.mllib.regression.LabeledPoint(0.77, org.apache.spark.mllib.linalg.Vectors.dense(0, 0, 1, 1, 0, 0)),
      new org.apache.spark.mllib.regression.LabeledPoint(1.43, org.apache.spark.mllib.linalg.Vectors.dense(0, 0, 1, 0, 1, 1)),
      new org.apache.spark.mllib.regression.LabeledPoint(1.48, org.apache.spark.mllib.linalg.Vectors.dense(0, 0, 1, 0, 1, 0))
*/
    ), 1)
    //data = data.repartition(1)
    data.take(10).foreach(println)
    val splits = data.randomSplit(Array[Double](0.8, 0.2), 1L)
    val training = splits(0)
    val testLP = splits(1)
    val testV = testLP.map { p => p.features }
    val numTraining: Long = training.count
    val numTest: Long = testV.count
    System.out.println("Training: " + numTraining + ", test: " + numTest)
    System.out.println("training sample")
    training.take(10).foreach(println)
    System.out.println("test sample")
    testV.take(10).foreach(println)
    //val fm = FMWithSGD.train(training, task = 0, numIterations = 100, dim = (true, true, 4), regParam = (0, 0, 0), initStd = 0.1, stepSize = 0.15, miniBatchFraction = 1.0)
    val fm = FMWithLBFGS.train(training, task = 0, numIterations = 100, dim = (true, true, 4), regParam = (0, 0, 0), initStd = 0.1, numCorrections = 5)
    /*val hdfs = FileSystem.get(sc.sparkContext.hadoopConfiguration)
    if (hdfs.exists(new Path("output"))) hdfs.delete(new Path("output"), true)
    MLUtils.saveAsLibSVMFile(training, "output")
    fm.save(sc.sparkContext, "outputfm")*/
    val predictionAndLabels = testLP.map { p => (
      //Math.round(
      fm.predict(p.features)
      //).toDouble
      , p.label)
    }
    println("validate: real rating => predicted rating")
    for (e <- predictionAndLabels.collect)
      println("real %.3f => predicted %.3f delta %.3f".format(e._2, e._1, (e._1 - e._2).abs))
    val RMSE = Math.sqrt(predictionAndLabels.map { case (v, p) => math.pow(v - p, 2) }.mean())
    val meanRating = training.map(p => p.label).mean()
    val baselineRMSE = Math.sqrt(testLP.map(p => Math.pow(p.label - meanRating, 2)).mean())
    println(f"model meanRating $meanRating%.3f baseline RMSE $baselineRMSE%.3f model RMSE $RMSE%.3f")
    val improvement = (baselineRMSE - RMSE) / baselineRMSE * 100
    println(f"model improves the baseline by $improvement%.3f percents")
    /*val metricsD: MulticlassMetrics = new MulticlassMetrics(predictionAndLabels)
    println("accuracy = %.3f".format(metricsD.accuracy))*/
  }
}
