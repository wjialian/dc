package com.jisanpang.dc.classification.feature

import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

// 特征选择（Feature Selection）指的是在特征向量中选择出那些“优秀”的特征，组成新的、更“精简”的特征向量的过程。
// 它在高维数据分析中十分常用，可以剔除掉“冗余”和“无关”的特征，提升学习器的性能。
/**
  * @Description: 卡方选择器
  * ChiSqSelector代表卡方特征选择。它对具有分类特征的标记数据进行操作。
  * ChiSqSelector使用卡方独立性检验来决定选择哪些功能。
  * 它支持五种选择方法：numTopFeatures，percentile，fpr，fdr，fwe：
  *
  * numTopFeatures根据卡方检验选择固定数量的Top特征。这类似于产生具有最大预测能力的特征。
  * percentile与numTopFeatures类似，但是选择所有功能的一部分而不是固定数量。
  * fpr选择p值低于阈值的所有特征，从而控制选择的误报率。
  * fdr使用Benjamini-Hochberg过程选择错误发现率低于阈值的所有特征。
  * fwe选择p值低于阈值的所有特征。阈值按1 / numFeatures缩放，从而控制选择的家庭式错误率。
  * 默认情况下，选择方法是numTopFeatures，TopFeatures的默认数量设置为50。
  * 可以使用setSelectorType选择选择方法。
  **/
object No3Selection {
  //
  def main(args: Array[String]) {
    No3Selection.method01()
  }

  // 卡方选择则是统计学上常用的一种有监督特征选择方法，它通过对特征和真实标签之间进行卡方检验，
  // 来判断该特征和真实标签的关联程度，进而确定是否对其进行选择。
  def method01(): Unit ={
    val spark = SparkSession.builder().master("local").appName("my App Name").getOrCreate()
    // 一个具有八个样本，三个特征维度的数据集，分别为性别，年龄，薪资。
    // 标签有1，0两种，分别代表护肤和不护肤
    val data01 = spark.createDataFrame(Seq(
      (1, Vectors.dense(0.0,  18.0, 4000.0), 1),
      (2, Vectors.dense(0.0,  19.0, 6000.0), 1),
      (3, Vectors.dense(0.0,  22.0, 5000.0), 1),
      (4, Vectors.dense(0.0,  28.0, 6000.0), 0),
      (5, Vectors.dense(1.0,  29.0, 7000.0), 0),
      (6, Vectors.dense(1.0,  26.0, 4000.0), 0),
      (7, Vectors.dense(1.0,  21.0, 5000.0), 0),
      (8, Vectors.dense(1.0,  19.0, 6000.0), 1)
    )).toDF("id", "features", "label")

    // 用卡方选择进行特征选择器的训练，我们可以通过setNumTopFeatures(int)方法设置和标签关联性最强的特征数
    val selector = new ChiSqSelector().
      setNumTopFeatures(1).
      setFeaturesCol("features").
      setLabelCol("label").
      setOutputCol("selected-feature")
    val selector_model = selector.fit(data01)
    val result = selector_model.transform(data01)
    // 可以看见，第三列特征被选出作为最有用的特征列
    result.show(false)
  }
}
