package com.jisanpang.dc.classification

import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.sql.SparkSession


object No2LogisticRegression {
  //
  def main(args: Array[String]) {
    No2LogisticRegression.method01()
  }

  def method01(): Unit = {
    val spark = SparkSession.builder().master("local").appName("my App Name").getOrCreate()
    //LabeledPoint在监督学习中常用来存储标签和特征，其中要求标签的类型是double，特征的类型是Vector。这里，先把莺尾花的分类进行变换，
    // "Iris-setosa"对应分类0，"Iris-versicolor"对应分类1，"Iris-virginica"对应分类2；然后获取莺尾花的4个特征，存储在Vector中。
    val data = spark.sparkContext.textFile("file:///Users/wjl/IdeaProjects/sparkformyself/src/main/resources/iris")
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(
        if (parts(4) == "Iris-setosa") 0.toDouble
        else if (parts(4) == "Iris-versicolor") 1.toDouble
        else 2.toDouble,
        Vectors.dense(parts(0).toDouble, parts(1).toDouble, parts(2).toDouble, parts(3).toDouble))
    }
    parsedData.foreach { x => println(x) }
    // 首先进行数据集的划分，这里划分60%的训练集和40%的测试集
    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)
    // 训练逻辑回归模型，用set的方法设置参数：
    val model = new LogisticRegressionWithLBFGS().setNumClasses(3).run(training)
    // 开始预测
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }
    // 预测结果
    predictionAndLabels.foreach(x => println(x))
    val metrics = new MulticlassMetrics(predictionAndLabels)
    println("Precision = " + metrics.accuracy)
  }

  case class Iris(features: org.apache.spark.ml.linalg.Vector, label: String)

}
