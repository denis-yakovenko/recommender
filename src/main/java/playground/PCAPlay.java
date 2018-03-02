package playground;

import org.apache.spark.sql.SparkSession;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.feature.PCA;
import org.apache.spark.ml.feature.PCAModel;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class PCAPlay {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder().master("local[*]").appName("CARS").getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");
        List<Row> data = Arrays.asList(
                RowFactory.create(Vectors.sparse(5, new int[]{1, 3}, new double[]{1.0, 7.0})),
                RowFactory.create(Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0)),
                RowFactory.create(Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0))
        );

        StructType schema = new StructType(new StructField[]{
                new StructField("features", new VectorUDT(), false, Metadata.empty()),
        });

        Dataset<Row> df = spark.createDataFrame(data, schema);

        PCAModel pca = new PCA()
                .setInputCol("features")
                .setOutputCol("pcaFeatures")
                .setK(3)
                .fit(df);

        Dataset<Row> result = pca.transform(df).toDF();
        result.show(false);
    }
}
