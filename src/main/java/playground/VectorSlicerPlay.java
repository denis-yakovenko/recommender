package playground;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.attribute.Attribute;
import org.apache.spark.ml.attribute.AttributeGroup;
import org.apache.spark.ml.attribute.NumericAttribute;
import org.apache.spark.ml.feature.VectorSlicer;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;

public class VectorSlicerPlay {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder().master("local[*]").appName("CARS").getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");

        Attribute[] attrs = {
                NumericAttribute.defaultAttr().withName("f1"),
                NumericAttribute.defaultAttr().withName("f2"),
                NumericAttribute.defaultAttr().withName("f3")
        };
        AttributeGroup group = new AttributeGroup("userFeatures", attrs);

        List<Row> data = Arrays.asList(
                RowFactory.create(Vectors.sparse(3, new int[]{0, 1}, new double[]{-2.0, 2.3})),
                RowFactory.create(Vectors.dense(-2.0, 2.3, 0.0))
        );

        Dataset<Row> dataset =
                spark.createDataFrame(data, (new StructType()).add(group.toStructField()));

        VectorSlicer vectorSlicer = new VectorSlicer()
                .setInputCol("userFeatures").setOutputCol("features");

        //vectorSlicer.setIndices(new int[]{1}).setNames(new String[]{"f3"});
        vectorSlicer.setIndices(new int[]{1, 2});
        //vectorSlicer.setNames(new String[]{"f2", "f3"});

        Dataset<Row> output = vectorSlicer.transform(dataset);
        output.show(false);
    }
}
