package main.scala.nicocu97

import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.linalg.Vectors
//import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.classification.kNN_IS.kNN_IS
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import java.io.{File, PrintWriter}
import scala.math.sqrt

object Script_ROS_KNN {
      def main(arg: Array[String]) {
      
      //Configuración básica
      val jobName = "KNN con ROS"

      //Spark Configuration
      //val spark = SparkSession.builder.appName("BigData2").getOrCreate()
      val conf = new SparkConf().setAppName(jobName)
      val sc = new SparkContext(conf)
     
      //Log level
      sc.setLogLevel("ERROR")
      
      //import spark.implicits._

      /************************
       * Lectura de parámetros
       ************************/
      if (arg.length != 2) {
        System.err.println("Especificar: <ratio de sobremuestreo en ROS> <Número de vecinos cercanos>")
        System.exit(-1)
      }
      
      val overRate = arg(0).toDouble
      val k = arg(1).toInt

      if (overRate <= 0.0 || overRate>1.0) {
        System.err.println("El ratio de sobremuestreo debe de estar en (0,1]")
        System.exit(-1)
      }
      
      if (k <= 0) {
        System.err.println("El número de vecinos debe de ser mayor que 0")
        System.exit(-1)
      }
      
      val t_global_start = System.nanoTime() //Inicio cronometrado global de ejecución del script
      
      /**********************
       * Carga de datos			*
       **********************/
      
      // Cargar conjunto de train
      val pathTrain = "/user/datasets/master/higgs/higgsMaster-Train.data" //cluster
      //val pathTrain = "/home/nico/Escritorio/bigData2/higgs/higgsMaster-Train.data"
      
      val rawDataTrain = sc.textFile(pathTrain)

      var train = rawDataTrain.map { line =>
        val array = line.split(",")
        val arrayDouble = array.map(f => f.toDouble)
        val featureVector = Vectors.dense(arrayDouble.init)
        val label = arrayDouble.last.toInt
        LabeledPoint(label, featureVector)
      }

      //var train_df = train.toDF()

     /*var train = rawDataTrain.map { line =>
        val array = line.split(",")
        val arrayDouble = array.map(f => f.toDouble)
        val featureVector = Vectors.dense(arrayDouble.init)
        val label = arrayDouble.last.toInt
        LabeledPoint(label, featureVector)
      }.persist*/


      // Salvarlos en RAM
      //train.first
      
      println("Cargado con exito \""+ pathTrain + "\", un total de " + train.count + " filas")

      
      // Cargar conjunto de test
      val pathTest = "/user/datasets/master/higgs/higgsMaster-Test.data" //Cluster
      //val pathTest = "/home/nico/Escritorio/bigData2/higgs/higgsMaster-Test.data"

      val rawDataTest = sc.textFile(pathTest)
      
      var test = rawDataTest.map { line =>
        val array = line.split(",")
        val arrayDouble = array.map(f => f.toDouble)
        val featureVector = Vectors.dense(arrayDouble.init)
        val label = arrayDouble.last.toInt
        LabeledPoint(label, featureVector)
      }

      //var test_df = test.toDF()

      //Salvarlos en RAM
      //test.first
      
      println("Cargado con exito \""+ pathTest + "\", un total de " + test.count + " filas")

      /*********************************
       * Preprocesamiento del dataset  *
       *********************************/
      println("Inicio de Preprocesamiento")
      val t_pre_start = System.nanoTime() 
      
      //Normalización del dataset
      /*val featuresTrain = train.map(_.features)
      val featuresTest = test.map(_.features)

      val summaryTrain = Statistics.colStats(featuresTrain)
      val train_mean = summaryTrain.mean(0)
      val train_std = sqrt(summaryTrain.variance(0))

      train_df = train_df.map{ line => (line.get(2).subtract(train_mean)).divide(train_std) }
      test = test.map{ line => (line.features-train_mean)/train_std }*/
      /*val scaler = new StandardScaler(withMean=true, withStd=true).fit(Vectors.dense(train_df("featureVector")))

      train("featureVector") = scaler.transform(Vectors.dense(train_df("featureVector")))
      test("featureVector") = scaler.transform(Vectors.dense(test_df("featureVector")))

      test = test.map { line =>
        LabeledPoint(line("label"),
                     line("featureVector"))
      }.persist*/

      val featuresTrain = train.map(_.features)

      val scaler = new StandardScaler().fit(featuresTrain)

      //train = scaler.transform(train_df)
      //test = scaler.transform(test_df)

      // Convertir los datasets de train y test a RDD
      train = train.map { line =>
        LabeledPoint(line.label,
                     scaler.transform(line.features))
      }.persist


      test = test.map { line =>
        LabeledPoint(line.label,
                     scaler.transform(line.features))
      }.persist

      // Aplicación de ROS 
      val trainROS = imbalanceSamplingMethods.ROS(train, overRate)
      
      val t_prepro = (System.nanoTime() - t_pre_start) / 1e+9
      
      /********************************************
       * Entrenamiento de modelos KNN 	*
       *******************************************/
      // Entrenamiento de modelo KNN.
      //  Empty categoricalFeaturesInfo indicates all features are continuous.
      var numClasses = 2
      val numFeatures = 28
      var categoricalFeaturesInfo = Map[Int, Int]()
      val numPartitions = 15
      val maxWeight = 1.0
      
      println("Entrenando KNN:")
      println(s"- Número de clases: $numClasses")
      println(s"Número de vecinos a considerar para clasificación: $k")
      
      val t_train_start = System.nanoTime()
  
      val modelKNN = new kNN_IS(trainROS, test, k, 2, numClasses,
                        numFeatures, numPartitions, numPartitions, -1, maxWeight).setup()
      val t_train = (System.nanoTime() - t_train_start)/1e+9

      /**************************
      * Evaluación del modelo 	*
      **************************/
      println("Evaluando modelo:")
      val t_test_start = System.nanoTime()
      
      //Evaluación train
      
      // Evaluar modelo sobre train y calcular error
      //val predsAndLabelsTrainKNN = modelKNN.calculatePredictedRightClassesFuzzy(trainROS)
      
      //Evaluación test
      
      // Evaluar modelo de test y calcular error
      val predsAndLabelsTestKNN = modelKNN.predict(sc)
      
      val t_test = (System.nanoTime() - t_test_start)/1e+9
      
      //Métricas train
      /*
      val metrics_train = new MulticlassMetrics(predsAndLabelsTrainKNN)
      
      val precision_train = metrics_train.precision
      val accuracy_train = metrics_train.accuracy
      val recall_train = metrics_train.recall
      val fMeasure_train = metrics_train.fMeasure
      val cm_train = metrics_train.confusionMatrix
      
      val tpr_train = cm_train.apply(0, 0) / predsAndLabelsTrainKNN.count().toDouble
      val tnr_train = cm_train.apply(1, 1) / predsAndLabelsTrainKNN.count().toDouble
      
      val tprxtnr_train = tpr_train*tnr_train*/
      
      //Métricas test
      val metrics_test = new MulticlassMetrics(predsAndLabelsTestKNN)
      
      val precision_test = metrics_test.precision
      val accuracy_test = metrics_test.accuracy
      val recall_test = metrics_test.recall
      val fMeasure_test = metrics_test.fMeasure
      val cm_test = metrics_test.confusionMatrix
      
      val tpr_test = cm_test.apply(0, 0) / predsAndLabelsTestKNN.count().toDouble
      val tnr_test = cm_test.apply(1, 1) / predsAndLabelsTestKNN.count().toDouble
      
      val tprxtnr_test = tpr_test*tnr_test

      val t_global = (System.nanoTime() - t_global_start)/1e+9 // Fin de cronometrado global
      
      //para cluster
      val writer = new PrintWriter(s"/home/x20620439/results_KNN-k=$k-ROS=$overRate.txt")
      writer.write(
        s"Métricas obtenidas sobre KNN con k $k, y ratio de sobremuestreo $overRate:\n" +
        "TEST:\n" +
        "Accuracy: " + accuracy_test + "\n"+
        "Precision: " + precision_test + "\n" +
        "Recall: " + recall_test + "\n" +
        "F-Measure: " + fMeasure_test + "\n" +
        "TPR: " + tpr_test + "\n" +
        "TNR: " + tnr_test + "\n" +
        "TPRxTNR: " + tprxtnr_test + "\n" +
          "Confusion Matrix " + cm_test + "\n" +
        "-------------------------------\n"+
         "Tiempo de preprocesamiento: "+t_prepro+" s\n" +
         "Tiempo de train: "+t_train+" s\n" +
         "Tiempo de evaluación: "+t_test+" s\n" +
         "Tiempo total: "+t_global+ " s\n" +
         "MODELO:\n" +
         modelKNN.toString()
      )
      writer.close()
     } 
}
