package com.jisanpang.dc.classification.feature

import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Row
import org.apache.spark.ml.{Pipeline, PipelineModel}
object No1Extract {
  def main(args: Array[String]) {
    No1Extract.method01()
    //No1Extract.method00()
    //No1Extract.method02()
    //No1Extract.method03()
  }

  def method00(): Unit = {
    val spark = SparkSession.builder().master("local").appName("my App Name").getOrCreate()
    // 训练集，每一行为一组数据，label为标签，features为特征数据
    val training = spark.createDataFrame(Seq(
      (1.0, Vectors.dense(0.0, 1.1, 0.1)),
      (0.0, Vectors.dense(2.0, 1.0, -1.0)),
      (0.0, Vectors.dense(2.0, 1.3, 1.0)),
      (1.0, Vectors.dense(0.0, 1.2, -0.5))
    )).toDF("label", "features")
    // 创建逻辑回归算法实例
    val lr = new LogisticRegression()
    val pipeline = new Pipeline().setStages(Array(lr))
    // 调用fit方法，使用训练集训练模型
    val model = pipeline.fit(training)
    // 测试集，用于对训练好的模型测试其准确性
    val test = spark.createDataFrame(Seq(
      (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
      (0.0, Vectors.dense(3.0, 2.0, -0.1)),
      (1.0, Vectors.dense(0.0, 2.2, -1.5))
    )).toDF("label", "features")
    // 调用transform方法，将一个DataFrame转换为另一个DataFrame，对测试集的特征数据进行预测
    val result = model.transform(test)
    // 展现结果，features为测试集特征数据，label为测试集实际标签，prediction为预测标签
    result.select("features", "label", "prediction").show(false)
//      .select("features", "label","probability", "prediction")
//      .collect()
//      .foreach { case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>
//        println(s"($features, $label) -> prob=$prob, prediction=$prediction")
//      }
  }
  // TF-IDF
  // ​词频－逆向文件频率（TF-IDF）是一种在文本挖掘中广泛使用的特征向量化方法，它可以体现一个文档中词语在语料库中的重要程度。
  // TF: HashingTF是一个Transformer，接收词条的集合然后把这些集合转化成固定长度的特征向量。这个算法在哈希的同时会统计各个词条的词频。
  // IDF: IDF是一个Estimator，在一个数据集上应用它的fit（）方法，产生一个IDFModel。 该IDFModel
  // 接收特征向量（由HashingTF产生），然后计算每一个词在文档中出现的频次。IDF会减少那些在语料库中出现频率较高的词的权重。
  def method01(): Unit ={
    val spark = SparkSession.builder().master("local").appName("my App Name").getOrCreate()
    val data01 = spark.createDataFrame(Seq(
      (1, "I heard about Spark and love Spark"),
      (2, "I wish Java and Scala could be better"),
      (3, "Now I am learning about the IDF")
             )).toDF("label", "sentence")
    // 用Tokenizer对句子进行分词
    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val data02 = tokenizer.transform(data01)
    data02.show(false)

    // 使用HashingTF的transform()方法把句子哈希成特征向量，这里设置哈希表的桶数为2000
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(2000)
    val data03 = hashingTF.transform(data02)
    // ​可以看出，分词序列被变换成一个稀疏特征向量，其中每个单词都被散列成了一个不同的向量值，向量值小于2000，
    // 特征向量在某一维度上的值即该词汇出现的次数。
    data03.select("words","rawFeatures").show(false)

    // 使用IDF来对单纯的词频特征向量进行修正，使其更能体现不同词汇对文本的区别能力
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    // IDF是一个Estimator，调用fit()方法并将词频向量传入，即产生一个IDFModel
    val idfModel = idf.fit(data03)
    // IDFModel是一个Transformer，调用它的transform()方法，即可得到每一个单词对应的TF-IDF度量值
    val data04 = idfModel.transform(data03)
    // 可以看到，特征向量已经被其在语料库中出现的总次数进行了修正，通过TF-IDF得到的特征向量，在接下来可以被应用到相关的机器学习方法中。
    data04.select("label","features").show(false)
  }

  // 如果词的语义相近，它们的词向量在向量空间中也相互接近，这使得词语的向量化建模更加精确，可以改善现有方法并提高鲁棒性。
  // 词向量已被证明在许多自然语言处理问题，如：机器翻译，标注问题，实体识别等问题中具有非常重要的作用。
  // ​ Word2vec是一个Estimator，它采用一系列代表文档的词语来训练word2vecmodel。该模型将每个词语映射到一个固定大小的向量。
  //  word2vecmodel使用文档中每个词语的平均数来将文档转换为向量，然后这个向量可以作为预测的特征，来计算文档相似度计算等等。
  def method02(): Unit ={
    val spark = SparkSession.builder().master("local").appName("my App Name").getOrCreate()
    // 创建三个词语序列，每个代表一个文档
    val documentDF = spark.createDataFrame(Seq(
      "Hi I heard about Spark".split(" "),
      "I wish Java could use case classes".split(" "),
      "Logistic regression models are neat".split(" ")
    ).map(Tuple1.apply)).toDF("text")

    // 新建一个Word2Vec，它是一个Estimator，设置相应的超参数，这里设置特征向量的维度为3
    val word2Vec = new Word2Vec().setInputCol("text").setOutputCol("result").setVectorSize(3).setMinCount(0)
    val model = word2Vec.fit(documentDF)
    val result = model.transform(documentDF)
    // 可以看到，文档被转变为了一个3维的特征向量
    result.show(false)
  }

  // CountVectorizer旨在通过计数来将一个文档转换为向量。
  // Countvectorizer作为Estimator提取词汇进行训练，并生成一个CountVectorizerModel用于存储相应的词汇向量空间。
  // 在CountVectorizerModel的训练过程中，CountVectorizer将根据语料库中的词频排序从高到低进行选择，
  def method03(): Unit ={
    val spark = SparkSession.builder().master("local").appName("my App Name").getOrCreate()
    // 假设我们有如下的DataFrame，其包含id和words两列，可以看成是一个包含两个文档的迷你语料库。
    val df = spark.createDataFrame(Seq(
      (0, Array("a", "b", "c")),
      (1, Array("a", "b", "b", "c", "a","b"))
    )).toDF("id", "words")
    // 通过CountVectorizer设定超参数，训练一个CountVectorizerModel，
    // 这里设定词汇表的最大量为3，设定词汇表中的词至少要在2个文档中出现过，以过滤那些偶然出现的词汇。
    val cvModel: CountVectorizerModel = new CountVectorizer().
      setInputCol("words").
      setOutputCol("features").
      setVocabSize(3).
      setMinDF(2).
      fit(df)
    // 通过CountVectorizerModel的vocabulary成员获得到模型的词汇表
    cvModel.vocabulary.foreach(println)
    // 文档的向量化表示
    cvModel.transform(df).show(false)
  }
}