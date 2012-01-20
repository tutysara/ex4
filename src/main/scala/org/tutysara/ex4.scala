package org.tutysara
import scalala.library.MATStorage._
import com.jmatio.types._
//scalala imports from wiki
import scalala.scalar._;
import scalala.tensor.::;
import scalala.tensor.mutable._;
import scalala.tensor.dense._;
import scalala.tensor.sparse._;
import scalala.library.Library._;
import scalala.library.LinearAlgebra._;
import scalala.library.Statistics._;
import scalala.library.Plotting._;
import scalala.operators.Implicits._;
import org.tutysara.Util.pause
import org.tutysara.Util.shuffle
/*
  Machine Learning Online Class - Exercise 4 Neural Network Learning


  Instructions
  ------------
 
  This file contains code that helps you get started on the
  linear exercise. You will need to complete the following functions 
  in this exericse:

     sigmoidGradient.m
     randInitializeWeights.m
     nnCostFunction.m

  For this exercise, you will not need to change any code in this file,
  or any other files other than those mentioned above.

*/

object ex4 {

  def main(args: Array[String]): Unit = {
    //Setup the parameters you will use for this exercise
val input_layer_size  = 400;  // 20x20 Input Images of Digits
val hidden_layer_size = 25;   //25 hidden unit
val num_labels = 10;          //10 labels, from 1 to 10   
    //(note that we have mapped "0" to label 10) //try changing this
/*
  =========== Part 1: Loading and Visualizing Data =============
  We start the exercise by first loading and visualizing the dataset. 
  You will be working with a dataset that contains handwritten digits.
*/

// Load Training Data
printf("Loading and Visualizing Data ...\n")

val FILE_DIR="/home/tutysra/testdata/ex4"
val FILE_DATA=FILE_DIR+"/ex4data1.mat"
val FILE_WEIGHTS=FILE_DIR+"/ex4weights.mat"
val varX=load(FILE_DATA,"X");
val X=varX.get.asInstanceOf[MLDouble].asMatrix //check and get

val m =X.numRows

//Randomly select 100 data points to display
val shuffled_idx = shuffle(0 until X.numRows toArray)
val X_sel=DenseMatrix.tabulate[Double](100, X.numCols)( 
			(i,j) =>X(shuffled_idx(i),j) //shuffle rows
			)
displayData(X_sel);

printf("Program paused. Press enter to continue.\n");
pause();

  }

}