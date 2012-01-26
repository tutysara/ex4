package org.tutysara

import com.jmatio.types._
import scalala.scalar._
import scalala.tensor.::
	import scalala.tensor.mutable._
	import scalala.tensor.dense._
	import scalala.tensor.sparse._
	import scalala.library.Library._
	import scalala.library.LinearAlgebra._
	import scalala.library.Statistics._
	import scalala.library.Plotting._
	import scalala.operators.Implicits._
	import scalala.library.Storage
	import org.tutysara.Util._
	import java.io.BufferedOutputStream
	import java.io.FileOutputStream
	import java.io.File
	import scalala.library.MATStorage._

	/*
  PREDICT Predict the label of an input given a trained neural network

   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
   trained weights of a neural network (Theta1, Theta2)
	 */
	object predict {

	def apply(Theta1: DenseMatrix[Double], Theta2: DenseMatrix[Double], X: DenseMatrix[Double]): DenseVectorCol[Int] = {
			// Useful values
			val m = X.numRows
			val num_labels = Theta2.numRows

			// You need to return the following variables correctly 
			val p = DenseVectorCol.zeros[Double](m)
			
			val h1 = sigmoid1(DenseMatrix.horzcat(DenseMatrix.ones[Double](X.numRows, 1), X) * Theta1.t);			
			val h2 = sigmoid1(DenseMatrix.horzcat(DenseMatrix.ones[Double](h1.numRows, 1), h1) * Theta2.t);			
			
			
			val res = maxindex(h2) 
			
			res:+=1//since index starts from 1 on octave and 0 is mapped to 10
			// =========================================================================
	}
	def main(args: Array[String]): Unit = {
			val FILE_DIR = "/home/tutysra/testdata/ex4"
				val FILE_DATA = FILE_DIR + "/ex4data1.mat"
				val FILE_THETA = FILE_DIR + "/theta_learned"
				val varMap1 = load(FILE_DATA, "X", "y");
			val X = varMap1.get("X").get.get.asInstanceOf[MLDouble].asMatrix
			val y = varMap1.get("y").get.get.asInstanceOf[MLDouble].asVector
			val varMap2 = load(FILE_THETA, "Theta1", "Theta2");
			println(varMap2)
			val Theta1_8_2 = varMap2.get("Theta1").get.get.asInstanceOf[MLDouble].asMatrix

			val Theta2_8_2 = varMap2 .get("Theta2").get.get.asInstanceOf[MLDouble].asMatrix
			val pred = predict(Theta1_8_2.toDense, Theta2_8_2.toDense, X.toDense);
			
			val pred_res = DenseVectorCol.tabulate[Double](pred.length)(
					(i) => if (y(i) == pred(i)) 1 else 0)
			println("pred_res = \n" + pred_res)
			var prediction_accuracy = mean(pred_res)*100
			printf("\nTraining Set Accuracy: %f\n", prediction_accuracy);
	}

}