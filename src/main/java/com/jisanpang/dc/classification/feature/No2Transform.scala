package com.jisanpang.dc.classification.feature

import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.Vectors

// 在机器学习处理过程中，为了方便相关算法的实现，经常需要把标签数据（一般是字符串）转化成整数索引，
// 或是在计算结束后将整数索引还原为相应的标签。
object No2Transform {
  //
  def main(args: Array[String]) {
    //No2Transform.method01()
    No2Transform.method02()
  }
  def method01(): Unit ={
    val spark = SparkSession.builder().master("local").appName("my App Name").getOrCreate()
    // StringIndexer转换器可以把一列类别型的特征（或标签）进行编码，使其数值化，索引的范围从0开始，
    // 该过程可以使得相应的特征索引化，使得某些无法接受类别型特征的算法可以使用，并提高诸如决策树等机器学习算法的效率。
    // 索引构建的顺序为标签的频率，优先编码频率较大的标签，所以出现频率最高的标签为0号。
    // 如果输入的是数值型的，我们会把它转化成字符型，然后再对其进行编码。
    val data01 = spark.createDataFrame(Seq(
      (0, "a"),(1, "b"),(2, "c"),(3, "a"),(4, "a"),(5, "c")))
      .toDF("id", "category")
    // 创建一个StringIndexer对象，设定输入输出列名，并对这个DataFrame进行训练，产生StringIndexerModel对象
    val indexer = new StringIndexer().setInputCol("category").setOutputCol("categoryIndex")
    val model = indexer.fit(data01)
    val data02 = model.transform(data01)
    // 可以看到，StringIndexerModel依次按照出现频率的高低，把字符标签进行了排序，即出现最多的“a”被编号成0，“c”为1，出现最少的“b”为2。
    data02.show()

    // ​与StringIndexer相对应，IndexToString的作用是把标签索引的一列重新映射回原有的字符型标签。
    // 主要使用场景一般都是和StringIndexer配合，先用StringIndexer将标签转化成标签索引，进行模型训练，
    // 然后在预测标签的时候再把标签索引转化成原有的字符标签
    val converter = new IndexToString().setInputCol("categoryIndex").setOutputCol("originalCategory")
    val data03 = converter.transform(data02)
    data03.show()
  }

  def method02(): Unit ={
    val spark = SparkSession.builder().master("local").appName("my App Name").getOrCreate()
    val data = Array(
      Vectors.dense(5.4, 4.4),
      Vectors.dense(-2.6, -1.6),
      Vectors.dense(-3.6, -2.6),
      Vectors.dense(2.4, 1.9),
      Vectors.dense(-1.6, -2.1)
    )
    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(1)
      .fit(df)

    val result = pca.transform(df)
    result.show(false)
  }
}
