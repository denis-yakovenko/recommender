package playground;

import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.apache.spark.sql.types.DataTypes.*;

public class Playground {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder().master("local[*]").appName("CARS").getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");

        /*
        Dataset<Row> df = spark
                .sqlContext()
                .createDataset(
                        JavaPairRDD.toRDD(predictionAndLabels1),
                        Encoders.tuple(
                                Encoders.DOUBLE(), Encoders.DOUBLE()
                        )
                )
                .toDF();*/

        StructType schema = createStructType(new StructField[]{
                createStructField("id", IntegerType, false),
                createStructField("hour", IntegerType, false),
                createStructField("mobile", DoubleType, false),
                createStructField("userFeatures", new VectorUDT(), false),
                createStructField("clicked", DoubleType, false)
        });
        Row row = RowFactory.create(0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0);
        Dataset<Row> dataset = spark.createDataFrame(Collections.singletonList(row), schema);

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"hour", "mobile"/*, "userFeatures"*/})
                .setOutputCol("features");

        Dataset<Row> output = assembler.transform(dataset);
        System.out.println("Assembled columns 'hour', 'mobile', 'userFeatures' to vector column " +
                "'features'");
        output.show(false);


        List<Row> data = Arrays.asList(
                RowFactory.create(0.0, "Hi I heard about Spark"),
                RowFactory.create(0.0, "I wish Java could use case classes"),
                RowFactory.create(1.0, "Logistic regression models are neat")
        );

        schema = new StructType(new StructField[]{
                new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("sentence", DataTypes.StringType, false, Metadata.empty())
        });
        Dataset<Row> sentenceData = spark.createDataFrame(data, schema);

        Tokenizer tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words");
        Dataset<Row> wordsData = tokenizer.transform(sentenceData);

        int numFeatures = 20;
        HashingTF hashingTF = new HashingTF()
                .setInputCol("words")
                .setOutputCol("rawFeatures")
                .setNumFeatures(numFeatures);

        Dataset<Row> featurizedData = hashingTF.transform(wordsData);
// alternatively, CountVectorizer can also be used to get term frequency vectors

        IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
        IDFModel idfModel = idf.fit(featurizedData);

        Dataset<Row> rescaledData = idfModel.transform(featurizedData);
        rescaledData.show(false);

// Input data: Each row is a bag of words from a sentence or document.
        data = Arrays.asList(
                RowFactory.create(Arrays.asList("Hi I heard about Spark".split(" "))),
                RowFactory.create(Arrays.asList("I wish Java could use case classes".split(" "))),
                RowFactory.create(Arrays.asList("Logistic regression models are neat".split(" ")))
        );
        schema = new StructType(new StructField[]{
                new StructField("text", new ArrayType(DataTypes.StringType, true), false, Metadata.empty())
        });
        Dataset<Row> documentDF = spark.createDataFrame(data, schema);

// Learn a mapping from words to Vectors.
        Word2Vec word2Vec = new Word2Vec()
                .setInputCol("text")
                .setOutputCol("result")
                .setVectorSize(3)
                .setMinCount(0);

        Word2VecModel model = word2Vec.fit(documentDF);
        Dataset<Row> result = model.transform(documentDF);
        result.show(false);

        for (Row row1 : result.collectAsList()) {
            List<String> text = row1.getList(0);
            Vector vector = (Vector) row1.get(1);
            System.out.println("Text: " + text + " => \nVector: " + vector + "\n");
        }

        data = Arrays.asList(
                RowFactory.create(Vectors.dense(0.0, 1.0, -2.0, 3.0)),
                RowFactory.create(Vectors.dense(-1.0, 2.0, 4.0, -7.0)),
                RowFactory.create(Vectors.dense(14.0, -2.0, -5.0, 1.0))
        );
        schema = new StructType(new StructField[]{
                new StructField("features", new VectorUDT(), false, Metadata.empty()),
        });
        Dataset<Row> df = spark.createDataFrame(data, schema);

        DCT dct = new DCT()
                .setInputCol("features")
                .setOutputCol("featuresDCT")
                .setInverse(false);

        Dataset<Row> dctDf = dct.transform(df);

        dctDf.show(false);

    }
}
