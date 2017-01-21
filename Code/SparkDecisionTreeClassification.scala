package com.sparkscala

import java.io.PrintWriter
import java.io.File
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.sql.Row
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.util.MLUtils
import scala.collection.mutable.WrappedArray
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.DecisionTreeClassificationModel

object SparkClassification {

  case class Article(id: Integer, text: String, cont_name: Integer)

  def main(args: Array[String]) {
    val xml = scala.xml.XML.loadFile(args(0))
    val pw = new PrintWriter(new File(args(1)))
    val mediawiki = (xml \\ "mediawiki" \\ "page").foreach {
      page =>
        var result = (page \ "id").text + "," +
          (page \ "revision" \ "text").text.toLowerCase().trim().replaceAll("[^A-Za-z ]+", " ").trim().replaceAll(" +", " ").toString() + ","
        if ((page \ "revision" \ "contributor" \ "id").text.length() > 0)
          result = result + (page \ "revision" \ "contributor" \ "id").text
        else
          result = result + "0"
        pw.println(result)
    }
    pw.close()

    val conf = new SparkConf().setAppName("SparkDecisionTree").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._

    def parseArticle(str: String): Article = {
      val line = str.split(",", -1)
      Article(line(0).toInt, line(1), line(2).toInt)
    }

    val textRDD = sc.textFile(args(1))

    val articleRDD = textRDD.map(parseArticle).cache()

    val dataset = articleRDD.toDF()

    dataset.show()
    /*
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")

    val wordsData = tokenizer.transform(dataset)

    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered")

    val filtereddata = remover.transform(wordsData)

    val wordcount = filtereddata.explode("filtered", "explode")((line: WrappedArray[String]) => line)

    import org.apache.spark.sql.functions._

    import org.apache.spark.sql.expressions.Window

    val count = wordcount.groupBy("filtered", "cont_name", "explode").count()

    val finalcount = Window.partitionBy(count("filtered")).orderBy(count("count").desc)

    val result = count.withColumn("max_count", first(count("count")).over(finalcount).as("max_count")).filter("count = max_count")

    val data = result.select("cont_name", "explode")

    sc.parallelize(data.select("cont_name", "explode").repartition(1).collect().toSeq).saveAsTextFile("result_med.txt")

    val assembler = new VectorAssembler().setInputCols(Array("cont_name")).setOutputCol("cont")

    val inputData = assembler.transform(data)

    val labelIndexer = new StringIndexer().setInputCol("explode").setOutputCol("indexedLabel").fit(inputData)

    val contCount = inputData.select("cont").distinct().count().toInt

    val featureIndexer = new VectorIndexer().setMaxCategories(contCount).setInputCol("cont").setOutputCol("indexedFeatures").fit(inputData)

    val Array(trainingData, testData) = inputData.randomSplit(Array(0.7, 0.3))

    val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxBins(contCount).setMaxDepth(5)

    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)

    sc.parallelize(predictions.select("predictedLabel", "explode", "cont").repartition(1).collect().toSeq).saveAsTextFile("prediction.txt")

    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)

    println("Accuracy = " + accuracy)

    println("Test Error = " + (1.0 - accuracy))

    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]

    println("Learned Classification tree model:\n" + treeModel.toDebugString)
*/
  }

}