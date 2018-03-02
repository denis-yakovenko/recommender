package playground;

import org.apache.spark.sql.SparkSession;
import java.util.Arrays;
import java.util.List;
/*import org.apache.spark.ml.feature.OneHotEncoderEstimator;
import org.apache.spark.ml.feature.OneHotEncoderModel;*/
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class OneHotEncoderEstimatorPlay {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder().master("local[*]").appName("CARS").getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");
        List<Row> data = Arrays.asList(
                RowFactory.create(0.0, 1.0),
                RowFactory.create(1.0, 0.0),
                RowFactory.create(2.0, 1.0),
                RowFactory.create(0.0, 2.0),
                RowFactory.create(0.0, 1.0),
                RowFactory.create(2.0, 0.0)
        );

        StructType schema = new StructType(new StructField[]{
                new StructField("categoryIndex1", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("categoryIndex2", DataTypes.DoubleType, false, Metadata.empty())
        });

        Dataset<Row> df = spark.createDataFrame(data, schema);

        /*OneHotEncoderEstimator encoder = new OneHotEncoderEstimator()
                .setInputCols(new String[] {"categoryIndex1", "categoryIndex2"})
                .setOutputCols(new String[] {"categoryVec1", "categoryVec2"});

        OneHotEncoderModel model = encoder.fit(df);
        Dataset<Row> encoded = model.transform(df);
        encoded.show();*/

    }
}
