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

object DocumentTF {

  case class Article(id: Integer, text: String)

  def main(args: Array[String]) {
    val xml = scala.xml.XML.loadFile(args(0))
    val pw = new PrintWriter(new File(args(1)))
    val mediawiki = (xml \\ "mediawiki" \\ "page").foreach {
      page =>
        var result = (page \ "id").text + "," + (page \ "revision" \ "text").text.toLowerCase().trim().replaceAll("[^A-Za-z ]+", " ").trim().replaceAll(" +", " ").toString()
        pw.println(result)
    }
    pw.close()

    val conf = new SparkConf().setAppName("SparkKMeans").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    def parseArticle(str: String): Article = {
      val line = str.split(",", -1)
      Article(line(0).toInt, line(1))
    }

    val textRDD = sc.textFile(args(1))

    val articleRDD = textRDD.map(parseArticle).cache()

    val dataset = articleRDD.toDF()
    
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val wordsData = tokenizer.transform(dataset)
    
    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered")

    val filtereddata = remover.transform(wordsData)

    val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol(remover.getOutputCol).setOutputCol("features")

    val featurizedData = hashingTF.transform(filtereddata)    
  
    val inverseDocumentFreq = new IDF().setInputCol(hashingTF.getOutputCol).setOutputCol("frequency")

    val idfOutput = inverseDocumentFreq.fit(featurizedData).transform(featurizedData)

    import org.apache.spark.ml.feature.Normalizer

    val normalizer = new Normalizer().setInputCol(inverseDocumentFreq.getOutputCol).setOutputCol("norm")

    val normOutput = normalizer.transform(idfOutput)
    
    normOutput.show()

    val km = new KMeans().setK(4).setSeed(43L).setFeaturesCol(normalizer.getOutputCol).setPredictionCol("prediction")

    val pipeline = new Pipeline().setStages(Array(tokenizer, remover, hashingTF, inverseDocumentFreq, normalizer, km))

    val pipelineModel = pipeline.fit(dataset)

    val pipelineResult = pipelineModel.transform(dataset)

    val finalResult = pipelineResult.select("id", "prediction")

    finalResult.repartition(1).write.format("com.databricks.spark.csv").option("header", true).save("prediction.csv")

    sc.parallelize(pipelineResult.repartition(1).collect().toSeq).saveAsTextFile("prediction.txt")
    
    pipelineResult.show()

  }

}