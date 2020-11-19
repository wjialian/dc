package com.jisanpang.dc.classification

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession


/**
  * ​决策树（decision tree）是一种基本的分类与回归方法，决策树模式呈树形结构，其中每个内部节点表示一个属性上的测试，
  * 每个分支代表一个测试输出，每个叶节点代表一种类别。学习时利用训练数据，根据损失函数最小化的原则建立决策树模型；
  * 预测时，对新的数据，利用决策树模型进行分类。
  * 决策树学习通常包括3个步骤：特征选择、决策树的生成和决策树的剪枝。
  */

/**
  * 特征选择在于选取对训练数据具有分类能力的特征，这样可以提高决策树学习的效率。通常特征选择的准则是信息增益（或信息增益比、基尼指数等）
  * ，每次计算每个特征的信息增益，并比较它们的大小，选择信息增益最大（信息增益比最大、基尼指数最小）的特征。
  * 特征选择的准则: 信息增益（informational entropy）表示得知某一特征后使得信息的不确定性减少的程度。
  */

/**
  * 决策树的生成
  * ​从根结点开始，对结点计算所有可能的特征的信息增益，选择信息增益最大的特征作为结点的特征
  * ，由该特征的不同取值建立子结点，再对子结点递归地调用以上方法，构建决策树；
  * 直到所有特征的信息增均很小或没有特征可以选择为止，最后得到一个决策树。
  *
  * ​决策树需要有停止条件来终止其生长的过程。一般来说最低的条件是：当该节点下面的所有记录都属于同一类，
  * 或者当所有的记录属性都具有相同的值时。这两种条件是停止决策树的必要条件，也是最低的条件。
  * 在实际运用中一般希望决策树提前停止生长，限定叶节点包含的最低数据量，以防止由于过度生长造成的过拟合问题。
  */

/**
  * 决策树的剪枝
  * ​决策树生成算法递归地产生决策树，直到不能继续下去为止。这样产生的树往往对训练数据的分类很准确，
  * 但对未知的测试数据的分类却没有那么准确，即出现过拟合现象。解决这个问题的办法是考虑决策树的复杂度，
  * 对已生成的决策树进行简化，这个过程称为剪枝。
  */
object No1DecisionTree {
  //
  def main(args: Array[String]) {
    No1DecisionTree.method01()
  }

  def method01(): Unit ={
    val spark = SparkSession.builder().master("local").appName("my App Name").getOrCreate()

    // ​导入spark.implicits._，使其支持把一个RDD隐式转换为一个DataFrame。
    // 我们用case class定义一个schema:Iris，Iris就是我们需要的数据的结构；
    // 然后读取文本文件，第一个map把每行的数据用“,”隔开，
    // 比如在我们的数据集中，每行被分成了5部分，前4部分是鸢尾花的4个特征，最后一部分是鸢尾花的分类；
    // 我们这里把特征存储在Vector中，创建一个Iris模式的RDD，然后转化成dataframe；
    // 然后把刚刚得到的数据注册成一个表iris，注册成这个表之后，我们就可以通过sql语句进行数据查询；
    // 选出我们需要的数据后，我们可以把结果打印出来查看一下数据。
    import spark.implicits._
    val data = spark.sparkContext.textFile("file:///Users/wjl/IdeaProjects/sparkformyself/src/main/resources/iris")
      .map(_.split(",")).map(
      p => Iris(Vectors.dense(p(0).toDouble,p(1).toDouble,p(2).toDouble, p(3).toDouble),p(4).toString())).toDF()

    // 把字符串型特征值进行数值化，出现频率最高的特征值为0号
    val labelIndexer = new StringIndexer().
      setInputCol("label").
      setOutputCol("indexedLabel").fit(data)
    // 把离散值特征进行编号为0～maxCategories-1，如果特征值不重复个数大于maxCategories，则该特征视为连续值，不会重新编号。
    val featureIndexer = new VectorIndexer().
      setInputCol("features").
      setOutputCol("indexedFeatures").
      setMaxCategories(4).fit(data)
    // 把预测的类别重新从数值型转化成字符型
    val labelConverter = new IndexToString().
      setInputCol("prediction").
      setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
    // 把数据集随机分成训练集和测试集，其中训练集占70%。
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
    // 构建决策树分类模型(默认使用cart算法)
    val dtClassifier = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
    val pipelinedClassifier = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dtClassifier, labelConverter))
    // 训练模型
    val modelClassifier = pipelinedClassifier.fit(trainingData)
    // 正式预测
    val predictionsClassifier = modelClassifier.transform(testData)
    predictionsClassifier.select("features", "label", "predictedLabel").show(30)
    // 评估决策树分类模型
    val evaluatorClassifier = new MulticlassClassificationEvaluator().
      setLabelCol("indexedLabel").
      setPredictionCol("prediction").
      setMetricName("accuracy")
    val accuracy = evaluatorClassifier.evaluate(predictionsClassifier)
    println("Test Error = " + (1.0 - accuracy))
    val treeModelClassifier = modelClassifier.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println("Learned classification tree model:\n" + treeModelClassifier.toDebugString)
  }

  case class Iris(features: org.apache.spark.ml.linalg.Vector, label: String)
}
