package main.scala.nicocu97

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import java.io.{File, PrintWriter}

object Script_ROS_RandomForest {
   def main(arg: Array[String]) {
      
      //Configuración básica
      val jobName = "Random Forest con ROS"

      //Spark Configuration
      val conf = new SparkConf().setAppName(jobName)
      val sc = new SparkContext(conf)
     
      //Log level
      sc.setLogLevel("ERROR")
      
      /************************
       * Lectura de parámetros
       ************************/
      if (arg.length != 5) {
        System.err.println("Especificar: <ratio de sobremuestreo en ROS> <Método de evaluación de impureza> <Número de árboles a generar> <Profundidad máxima del árbol> <Número máximo de bins>")
        System.exit(-1)
      }
      
      val overRate = arg(0).toDouble
      val impurity = arg(1)
      val numTrees = arg(2).toInt
      val maxDepth = arg(3).toInt
      val maxBins = arg(4).toInt
      val featureSubsetStrategy = "auto" 

      if (overRate <= 0.0 || overRate>1.0) {
        System.err.println("El ratio de sobremuestreo debe de estar en (0,1]")
        System.exit(-1)
      }

      if (numTrees <= 0) {
        System.err.println("El número máximo de árboles debe ser mayor que 0")
        System.exit(-1)
      }
      
      if (maxDepth <= 0) {
        System.err.println("La profundidad máxima debe de ser mayor que 0")
        System.exit(-1)
      }
      
      if (maxBins <= 0) {
        System.err.println("El número máximo de bins debe ser mayor que 0")
        System.exit(-1)
      }
      
       val t_global_start = System.nanoTime() //Inicio cronometrado global de ejecución del script
      
      /**********************
       * Carga de datos			*
       **********************/
      
      // Cargar conjunto de train
      val pathTrain = "/user/datasets/master/higgs/higgsMaster-Train.data"
      
      val rawDataTrain = sc.textFile(pathTrain)
    
      val train = rawDataTrain.map { line =>
        val array = line.split(",")
        val arrayDouble = array.map(f => f.toDouble)
        val featureVector = Vectors.dense(arrayDouble.init)
        val label = arrayDouble.last.toInt
        LabeledPoint(label, featureVector)
      }.persist

      // Salvarlos en RAM

      
      println("Cargado con exito \""+ pathTrain + "\", un total de " + train.count + " filas")

      
      // Cargar conjunto de test
      val pathTest = "/user/datasets/master/higgs/higgsMaster-Test.data"

      val rawDataTest = sc.textFile(pathTest)
      
      val test = rawDataTest.map { line =>
        val array = line.split(",")
        val arrayDouble = array.map(f => f.toDouble)
        val featureVector = Vectors.dense(arrayDouble.init)
        val label = arrayDouble.last.toInt
        LabeledPoint(label, featureVector)
      }.persist
      
      //Salvarlos en RAM
      
      println("Cargado con exito \""+ pathTest + "\", un total de " + test.count + " filas")

      /*********************************
       * Preprocesamiento del dataset  *
       *********************************/
      println("Inicio de Preprocesamiento")
      val t_pre_start = System.nanoTime() 
      
      // Aplicación de ROS 
      val trainROS = imbalanceSamplingMethods.ROS(train, overRate)
      
      trainROS.persist()
      trainROS.first()
      
      val t_prepro = (System.nanoTime() - t_pre_start) / 1e+9
      
      /********************************************
       * Entrenamiento de modelos Random Forest 	*
       *******************************************/
      // Entrenamiento de modelo Random Forest.
      //  Empty categoricalFeaturesInfo indicates all features are continuous.
      var numClasses = 2
      var categoricalFeaturesInfo = Map[Int, Int]()
      
      println("Entrenando Decision Tree:")
      println(s"- Número de clases: $numClasses")
      println(s"- Médida de pureza: $impurity")
      println(s"Profundidad máxima: $maxDepth")
      println(s"Número máximo de particiones: $maxBins")
      println(s"Número máximo de árboles: $numTrees")
      
      val t_train_start = System.nanoTime()
  
      val modelRF = RandomForest.trainClassifier(trainROS, numClasses, categoricalFeaturesInfo, numTrees,
                                                  featureSubsetStrategy, impurity, maxDepth, maxBins)
  
      val t_train = (System.nanoTime() - t_train_start)/1e+9

      /**************************
      * Evaluación del modelo 	*
      **************************/

      println("Evaluando modelo:")
      val t_test_start = System.nanoTime()
      
      //Evaluación train
      
      // Evaluar modelo sobre train y calcular error
      val predsAndLabelsTrainRF = train.map { point =>
        val prediction = modelRF.predict(point.features)
        (prediction, point.label)
      }
      
      //Evaluación test
      
      // Evaluar modelo de test y calcular error
      val predsAndLabelsTestRF = test.map { point =>
        val prediction = modelRF.predict(point.features)
        (prediction, point.label)
      }
      
      val t_test = (System.nanoTime() - t_test_start)/1e+9
      
      //Métricas train
      val metrics_train = new MulticlassMetrics(predsAndLabelsTrainRF)
      
      val precision_train = metrics_train.precision
      val accuracy_train = metrics_train.accuracy
      val recall_train = metrics_train.recall
      val fMeasure_train = metrics_train.fMeasure
      val cm_train = metrics_train.confusionMatrix
      
      val tpr_train = cm_train.apply(0, 0) / predsAndLabelsTrainRF.count().toDouble
      val tnr_train = cm_train.apply(1, 1) / predsAndLabelsTrainRF.count().toDouble
      
      val tprxtnr_train = tpr_train*tnr_train
      
      //Métricas test
      val metrics_test = new MulticlassMetrics(predsAndLabelsTestRF)
      
      val precision_test = metrics_test.precision
      val accuracy_test = metrics_test.accuracy
      val recall_test = metrics_test.recall
      val fMeasure_test = metrics_test.fMeasure
      val cm_test = metrics_test.confusionMatrix
      
      val tpr_test = cm_test.apply(0, 0) / predsAndLabelsTestRF.count().toDouble
      val tnr_test = cm_test.apply(1, 1) / predsAndLabelsTestRF.count().toDouble
      
      val tprxtnr_test = tpr_test*tnr_test
      
      val t_global = (System.nanoTime() - t_global_start)/1e+9 // Fin de cronometrado global

      //para cluster
      val writer = new PrintWriter(s"/home/x20620439/results_RandomForest_impurity=$impurity-maxdepth=$maxDepth-max_Bins=$maxBins-numTrees=$numTrees-ROS=$overRate.txt")
      writer.write(
        s"Métricas obtenidas sobre RandomForest con métrica de impureza '$impurity', profunidad máxima $maxDepth, número de bins máximos $maxBins, número de árboles $numTrees y ratio de sobremuestreo $overRate:\n" +
        "TRAIN:\n" +
        "Accuracy: " + accuracy_train + "\n"+
        "Precision: " + precision_train + "\n" +
        "Recall: " + recall_train + "\n" +
        "F-Measure: " + fMeasure_train + "\n" +
        "TPR: " + tpr_train + "\n" +
        "TNR: " + tnr_train + "\n" +
        "TPRxTNR: " + tprxtnr_train + "\n" +
        "Confusion Matrix " + cm_train + "\n" +
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
         modelRF.toString()
      )
      writer.close()
     } 
}
