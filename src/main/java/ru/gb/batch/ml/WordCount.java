package ru.gb.batch.ml;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.api.java.JavaSparkContext;

import org.apache.log4j.Logger;
import org.apache.log4j.Level;

import org.apache.spark.sql.types.DataTypes;
import static org.apache.spark.sql.functions.*;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.Tokenizer;

import java.util.*;
import java.util.stream.Collectors;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Класс запускает Spark ML задачу, которая:
 * 1. зарускает на мастере args[0]
 * 2. Тренирует модель на датасете args[1]
 * 3. Тестирует модель на датасете args[2]
 */
public class WordCount {

    /**
     * Входная точка приложения. Считает количество слов во входном файле и пишет результат в выходной.
     */
    public static void main(String[] args) throws IOException {
        // проверка аргументов
        if (args.length < 1) {
            throw new IllegalArgumentException("Expected arguments: master [training-set] [validation-set]");
        }
        final String master = args.length > 0 ? args[0] : "local[*]";
        final String train  = args.length > 1 ? args[1] : "imdb/Train.csv";
        final String valid  = args.length > 2 ? args[2] : "imdb/Valid.csv";

        System.out.printf("My arguments: %s, %s, %s %n", master, train, valid); 

        // инициализация Spark
        SparkSession sqlc = SparkSession.builder().appName("IMDB sentiment analysis").master(master).getOrCreate();

        // увеличиваем порог уровня ошибок
        Logger.getRootLogger().setLevel(Level.ERROR);

        // читаем тренировочный датасет
        final Dataset<Row> dtrain = sqlc.read().option("header", "true").option("quote", "\"").option("escape", "\"").csv(train)
                                               .withColumn("label", col("label").cast(DataTypes.FloatType));

        // готовим конвейер
        Tokenizer tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words");
  
        HashingTF hashingTF = new HashingTF().setNumFeatures(1000).setInputCol(tokenizer.getOutputCol()).setOutputCol("features");
  
        RandomForestClassifier rf = new RandomForestClassifier().setFeaturesCol("features").setLabelCol("label");
  
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{tokenizer, hashingTF, rf});

        // запускаем обучение
        PipelineModel model = pipeline.fit(dtrain);

        // читаем тестовый датасет
        Dataset<Row> dtest = sqlc.read().option("header", "true").option("quote", "\"").option("escape", "\"").csv(valid)
                                              .withColumn("label", col("label").cast(DataTypes.FloatType));
        
        // запускаем предсказания
        Dataset<Row> dp = model.transform(dtest);

        // показываем сэмпл предсказаний
        dp.select("*").show(10);

        // проверяем точность модели
        List<Row>  rawResults = dp.withColumn("right", col("prediction").equalTo(col("label"))).groupBy("right").count().collectAsList();

        Map<Boolean, Long> results = rawResults.stream().collect(Collectors.toMap(row -> row.getBoolean(0), row -> row.getLong(1)));

        final double accuracy = 1.0*results.get(true)/(results.get(true)+results.get(false));

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy");

        final double eaccuracy = evaluator.evaluate(dp);

        // показываем результаты
       System.out.println(" RandomForestClassification ");
       System.out.printf(" Right prediction: %d%n Wrong prediction: %d%n Model accuracy: %.4f%n Accuracy evaluation: %.4f%n",results.get(true), results.get(false), accuracy, eaccuracy);

        // завершаем работу
        sqlc.stop();
    }

}
