package main.scala.nicocu97


import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint

object imbalanceSamplingMethods {
  // ROS
  def ROS(train: RDD[LabeledPoint], overRate: Double): RDD[LabeledPoint] = {
    var oversample: RDD[LabeledPoint] = train.sparkContext.emptyRDD
  
    val train_positive = train.filter(_.label == 1)
    val train_negative = train.filter(_.label == 0)
    val num_neg = train_negative.count().toDouble
    val num_pos = train_positive.count().toDouble
  
    if (num_pos > num_neg) {
      val fraction = (num_pos * overRate) / num_neg
      oversample = train_positive.union(train_negative.sample(withReplacement = true, fraction))
    } else {
      val fraction = (num_neg * overRate) / num_pos
      oversample = train_negative.union(train_positive.sample(withReplacement = true, fraction))
    }
    oversample.repartition(train.getNumPartitions)
  }


  // RUS
  
  def RUS(train: RDD[LabeledPoint]): RDD[LabeledPoint] = {
    var undersample: RDD[LabeledPoint] = train.sparkContext.emptyRDD
  
    val train_positive = train.filter(_.label == 1)
    val train_negative = train.filter(_.label == 0)
    val num_neg = train_negative.count().toDouble
    val num_pos = train_positive.count().toDouble
  
    if (num_pos > num_neg) {
      val fraction = num_neg / num_pos
      undersample = train_negative.union(train_positive.sample(withReplacement = false, fraction))
    } else {
      val fraction = num_pos / num_neg
      undersample = train_positive.union(train_negative.sample(withReplacement = false, fraction))
    }
    undersample
  }
}