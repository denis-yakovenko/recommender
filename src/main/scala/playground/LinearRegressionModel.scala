package playground

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

object LinearRegressionModel {

  def main(args: Array[String]): Unit = {
    val sc = SparkSession.builder.master("local[*]").appName("CARS").getOrCreate
    sc.sparkContext.setLogLevel("ERROR")
    val df = sc.createDataFrame(sc.sparkContext.parallelize(List(
      LabeledPoint(0.90, Vectors.dense(1, 0, 0, 1, 0, 1)),
      LabeledPoint(0.80, Vectors.dense(1, 0, 0, 1, 0, 0)),
      LabeledPoint(0.60, Vectors.dense(1, 0, 0, 0, 1, 1)),
      LabeledPoint(0.65, Vectors.dense(1, 0, 0, 0, 1, 0)),
      LabeledPoint(0.85, Vectors.dense(0, 1, 0, 1, 0, 1)),
      LabeledPoint(0.80, Vectors.dense(0, 1, 0, 1, 0, 0)),
      LabeledPoint(0.55, Vectors.dense(0, 1, 0, 0, 1, 1)),
      LabeledPoint(0.50, Vectors.dense(0, 1, 0, 0, 1, 0)),
      LabeledPoint(0.77, Vectors.dense(0, 0, 1, 1, 0, 1)),
      LabeledPoint(0.77, Vectors.dense(0, 0, 1, 1, 0, 0)),
      LabeledPoint(0.43, Vectors.dense(0, 0, 1, 0, 1, 1)),
      LabeledPoint(0.48, Vectors.dense(0, 0, 1, 0, 1, 0))), 1).map { row => LabeledPoint(row.label, row.features) })
    val splits = df.randomSplit(Array[Double](0.8, 0.2))
    splits(0).show(false)
    splits(1).show(false)
    val lr = new LinearRegression
    // Importing LinearRegressionModel and being explicit about the type of model value
    // is for learning purposes only
    val model = lr.fit(splits(0))
    // Use the same ds - just for learning purposes
    model.transform(splits(1)).show(false)
  }
}
