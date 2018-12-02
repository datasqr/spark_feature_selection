package com.boruta

import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{ SaveMode, SparkSession }
import org.apache.spark.sql.{ DataFrame, Row, SQLContext }

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorAssembler

import org.apache.spark.ml.regression.{ RandomForestRegressionModel, RandomForestRegressor }
import org.apache.spark.ml.evaluation.RegressionEvaluator

import org.apache.spark.sql.types._
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.sql.functions._
import scala.util.Random._
import scala.reflect.ClassTag
import scala.io.Source
import scala.util.control.Breaks._

import org.apache.commons.math3.distribution.NormalDistribution

/*
 * This example describes using RandomForestGergression
 * with max(Shadow feature importance) calculations
 * 
 * Question: If shuffle should of the shadow features should be every iteration?
 * Question: If split of data to train and test set should be every iteration?
 * 
 */

object FeatureSelectionBorutaZScore {

  def main(args: Array[String]): Unit = {

    val sparkConf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("Spark AnomalyDetection")
      .set("spark.driver.allowMultipleContexts", "true")

    val sc = new SparkContext(sparkConf)

    val sqlContext = new SQLContext(sc)

    val rdd = sc.textFile("/home/mz/DataInception/EY/Orlen_blending/data/dataTechnolodzy/tempdest_dataset_laby_czwarta.csv")

    val data = rdd.map(_.split(","))
      .map(_.map(_.toDouble))
      .map(x => x.drop(1)) //drop timestamp

    /*
     * Entering data should be in a form of RDD[Array(Double)]
     * The least element of the array is label - criterion variable (y)
     * The rest of the elements are predictor variables
     */

    val headers = sc.textFile("/home/mz/DataInception/EY/Orlen_blending/data/dataTechnolodzy/headers")

    val selectedeHeaderNames = headers.map(_.split(","))
      .map(_.map(_.toString))
      .map(x => x.drop(1).toList.zipWithIndex)

      println("selectedeHeaderNames")
    selectedeHeaderNames.foreach(println)

    val feat = selectFeatures(sqlContext, data)
      .toList

    println("result")
    feat.mkString(",")

    val selectedeHeaderNames1 = selectedeHeaderNames
      .filter(x => feat.contains(x.map(y => y._2)))

    println("selectedeHeaderNames1")
    selectedeHeaderNames1.foreach(println)

  }

  def selectFeatures(sqlContext: SQLContext, data: RDD[Array[Double]]): Array[Int] = {

    /*
     * The procedure to change columns to rows -> RDD[Seq[(Int, Double)]]
     * Key is the number of column
     */
    
    println("byColumnAndRow")

    val byColumnAndRow = data.zipWithIndex.flatMap {
      case (row, rowIndex) => row.zipWithIndex.map {
        case (number, columnIndex) => columnIndex -> (rowIndex, columnIndex + 1, number)
      }
    }

    // Build up the transposed matrix. Group and sort by column index first.
    val byColumn = byColumnAndRow.groupByKey.sortByKey().values

    // Then sort by row index.
    val transposed = byColumn.map {
      indexedRow => (indexedRow.toSeq.sortBy(_._1).map(x => (x._2, x._3)))
    }

    // Initialization of column names
    var columnsNamesPrime = transposed
      .map(x => x.map(y => y._1))
      .map(x => x(0))
      .collect()

    println("shadowColumnMultiplicator")
    // Value of the last column required to create names for shadow variables
    val numberOfColumns = columnsNamesPrime.max
    val columnNamesMultiplication = shadowColumnMultiplicator(numberOfColumns)

    var proceed = true
    var indexWhile = 0

    while (proceed) {
      
      println("IndexWhile " + indexWhile)

      /*
     * Create a copy of independent variables and randomly shuffle data in a columns
     */

      println("shuffled")
      val shuffled = transposed
        .filter(x => columnsNamesPrime.contains(x(0)._1))
        .map(x => Seq(x, scala.util.Random.shuffle(x.map(x => (x._1 * 100, x._2)))))

      // Bring back columns from rows
      val transpose = createColumnFromRow1(shuffled)

      //Remover the last column. The last column is shadow label (shuffled). We do not need it
      val transpose1 = transposeRdd(transpose)
        .map(x => x.dropRight(1))

      // Names of the columns
      val namesArray = transpose1.take(1)(0)
        .map(x => x._1.toString())
        .toArray
        .dropRight(1)

      // Add to the names of columns "lable"
      val namesFinal = namesArray :+ "label"

      // Create schema for dataframe
      val schema = StructType(namesFinal.map(x => StructField(x, DoubleType)))

      val df = transpose1
        .map(x => x.map(_._2))

      val df1 = df
        .map(Row.fromSeq(_))

      println("create data frame")
      // create data frame
      import sqlContext.implicits._
      val dataFrame = sqlContext.createDataFrame(df1, schema)

      // Prepare data frame for RF model
      val assembler = new VectorAssembler().setInputCols(namesArray.dropRight(1)).setOutputCol("features")
      val df4 = assembler.transform(dataFrame)

      // Divide data to training and test set
      val splitSeed = 5043
      val Array(trainingData, testData) = df4.randomSplit(Array(0.7, 0.3), splitSeed)

      // initialize array where values of features importance will be stored
      var selectImportantFeature: Array[(Int, Double)] = Array()

      val numberOfIterations = 10

      for (j <- 1 to numberOfIterations) {

        println("Number of Iterations" + j)
        // Run the model
        val classifier = new RandomForestRegressor() // RandomForestClassifier
          .setMaxDepth(30)
          .setNumTrees(500)
          .setFeatureSubsetStrategy("auto")
        //.setSeed(5043)

        val model = classifier.fit(trainingData)

        val predictions = model.transform(testData)

        val evaluator = new RegressionEvaluator() // MulticlassClassificationEvaluator
          .setLabelCol("label")
          .setPredictionCol("prediction")
        //.setMetricName("precision")

        val accuracy = evaluator.evaluate(predictions)

        // Get values of feature importance
        val importance = model
          .featureImportances

        val features = dataFrame.columns

        // Create array where values 
        val res = features.zip(importance.toArray).map(x => (x._1.toInt, x._2)).sortBy(-_._2)
        
        println("importance")
        res.foreach(println)

        /*
       * Select features which have higher value of importance than max value of shadow features importance 
       */

        val maxImportancForShadowFeature = res
          .filter(x => x._1 > columnNamesMultiplication + 1)
          .map(_._2)
          .max

        val selectImportantFeatureTemp = res
          .filter(x => x._1 < columnNamesMultiplication + 1)
          .filter(x => x._2 > maxImportancForShadowFeature)

        // store results in a array
        selectImportantFeature = selectImportantFeature ++ selectImportantFeatureTemp

      }

      println("calculate pValue")
      // Select important values
      val selectTheBestFeatures = selectImportantFeature
        .groupBy(_._1)
        .map(x => (x._1, (Array.fill(numberOfIterations - x._2.size)(0) ++ Array.fill(x._2.size)(1))))
        .map(x => (x._1, zScore(x._2.map(_.toDouble).toSeq)))
        .map(x => (x._1, x._2.maxBy(y => math.abs(y))))
        .map(x => (x._1, pValueCalculation(x._2)))
        .map(x => (x._1, if (x._2 < 0.01) "important" else "unimportant"))

      println("selectTheBestFeatures")
      selectTheBestFeatures.foreach(println)

      val numberOfValuesInResults = selectTheBestFeatures
        .keys
        .size

      val numberOfImportantValues = selectTheBestFeatures
        .count(x => x._2 == "important")

      val columnsNamesPrime1 = selectTheBestFeatures
        .filter(x => x._2 == "important")
        .map(x => x._1)
        .toArray

      columnsNamesPrime = columnsNamesPrime1

      println("columnsNamesPrime ")
      columnsNamesPrime1.foreach(println)

      // if there will be only important variables in break the loop
      if (numberOfValuesInResults == numberOfImportantValues) {
        proceed = false
        println("Proceed " + proceed)
        columnsNamesPrime = columnsNamesPrime1
        break
        return columnsNamesPrime1

      }
      
      indexWhile = indexWhile+1
      
    }

    columnsNamesPrime

  }
  
  def mapToKeyValuePairs(rdd: RDD[Array[Double]]): RDD[Seq[(Int,Double)]] = {
    val rowLength = rdd.take(0).size - 1
    val temp = rdd
      .flatMap(row => Range(0, rowLength).map(i => (i, row.indexOf(i).toDouble)))
      .groupBy(_._1)
      .map(_._2.toSeq)
    
    temp
  }
  
  def pValueCalculation(zScore: Double): Double = {
    
    //https://www.programcreek.com/java-api-examples/index.php?api=org.apache.commons.math3.distribution.NormalDistribution

    val standardNormal = new NormalDistribution(0, 1);

    val zTemp = math.abs(zScore)
      
    val pValue = 2 * standardNormal.cumulativeProbability(-zTemp);
    
    pValue

  }

  def shadowColumnMultiplicator(numCol: Int): Int = {

    // find next power of 10 larger than number
    numCol * (math.ceil(math.log10(numCol))).toInt

  }
  

  def transposeRdd(inputRdd: RDD[Seq[(Int, Double)]]): RDD[Seq[(Int, Double)]] = {

    // transpose: https://stackoverflow.com/questions/29390717/how-to-transpose-an-rdd-in-spark

    val byColumnAndRow = inputRdd.zipWithIndex.flatMap {
      case (row, rowIndex) => row.zipWithIndex.map {
        case (number, columnIndex) => columnIndex -> (rowIndex, number)
      }
    }

    // Build up the transposed matrix. Group and sort by column index first.
    val byColumn = byColumnAndRow.groupByKey.sortByKey().values

    //println("byColumn")
    //byColumn.take(10).foreach(println)

    // Then sort by row index.
    val transposed = byColumn.map {
      indexedRow => (indexedRow.toSeq.sortBy(_._1).map(_._2))
    }

    transposed

  }

  def createColumnFromRow1(rdd: RDD[Seq[Seq[(Int, Double)]]]): RDD[Seq[(Int, Double)]] = {

    //val rdd1 = sc.parallelize(Seq(Seq(Seq(1, 2, 3), Seq(4, 5, 6)), Seq(Seq(4, 5, 6), Seq(7, 8, 9))))
    // Split the matrix into one number per line.
    val byColumnAndRow = rdd.zipWithIndex.flatMap {
      case (row, rowIndex) => row.zipWithIndex.map {
        case (number, columnIndex) => columnIndex -> (rowIndex, number)
      }
    }

    val columnRdd = byColumnAndRow
      .map(x => x._2._2)

    columnRdd

  }

  def createColumnFromRow[A](rdd: RDD[Seq[A]])(implicit c: ClassTag[A]) = {

    //val rdd1 = sc.parallelize(Seq(Seq(Seq(1, 2, 3), Seq(4, 5, 6)), Seq(Seq(4, 5, 6), Seq(7, 8, 9))))
    // Split the matrix into one number per line.
    val byColumnAndRow = rdd.zipWithIndex.flatMap {
      case (row, rowIndex) => row.zipWithIndex.map {
        case (number, columnIndex) => columnIndex -> (rowIndex, number)
      }
    }

    val columnRdd = byColumnAndRow
      .map(x => x._2._2)

    columnRdd

  }

  def mean[T](item: Traversable[T])(implicit n: Numeric[T]) = {
    n.toDouble(item.sum) / item.size.toDouble
  }

  def variance[T](items: Traversable[T])(implicit n: Numeric[T]): Double = {
    val itemMean = mean(items)
    val count = items.size
    val sumOfSquares = items.foldLeft(0.0d)((total, item) => {
      val itemDbl = n.toDouble(item)
      val square = math.pow(itemDbl - itemMean, 2)
      total + square
    })
    sumOfSquares / count.toDouble
  }

  def stddev[T](items: Traversable[T])(implicit n: Numeric[T]): Double = {
    math.sqrt(variance(items))
  }

  def zScore(items: Seq[(Double)]): Seq[(Double)] = {

    val temp = items

    val meanVal = mean(temp)

    val std = stddev(temp)

    var zscore = ArrayBuffer[Double]()
    for (item <- temp) {
      zscore += (item - mean(temp)) / stddev(temp)
    }

    val zscore1 = items
      .map(x => (x - mean(temp)) / stddev(temp))

    zscore1

  }

}