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
val varMap=load(FILE_DATA,"X","y");
val X=varMap.get("X").get.get.asInstanceOf[MLDouble].asMatrix //check and get
val y=varMap.get("y").get.get.asInstanceOf[MLDouble].asVector //check and get
println("y="+y)
val m =X.numRows

//Randomly select 100 data points to display
val shuffled_idx = shuffle(0 until X.numRows toArray)
val X_sel=DenseMatrix.tabulate[Double](100, X.numCols)( 
			(i,j) =>X(shuffled_idx(i),j) //shuffle rows
			)
displayData(X_sel);

printf("Program paused. Press enter to continue.\n");
//pause();
/* 
 ================ Part 2: Loading Pameters ================
 In this part of the exercise, we load some pre-initialized 
 neural network parameters.
*/
printf("\nLoading Saved Neural Network Parameters ...\n")

// Load the weights into variables Theta1 and Theta2
val varMap_weights=load(FILE_WEIGHTS,"Theta1","Theta2");
val Theta1=varMap_weights.get("Theta1").get.get.asInstanceOf[MLDouble].asMatrix
val Theta2=varMap_weights.get("Theta2").get.get.asInstanceOf[MLDouble].asMatrix
	println("Theta1 and Theta2 patches when they are loaded")
	println("Theta1 = "+Theta1.numRows,Theta1.numCols)
	println("Theta2 = "+Theta2.numRows,Theta2.numCols)
	//println("Theta1 patch = \n"+Theta1(0 to 5,0 to 5))
	println("Theta2 patch 1= \n"+Theta2(0 to 9,0 to 7))
	println("Theta2 patch 2= \n"+Theta2(0 to 9,8 to 15))
	println("Theta2 patch 3= \n"+Theta2(0 to 9,16 to 23))
	println("Theta2 patch 3= \n"+Theta2(0 to 9,24 to 25))
	
	//println("Theta2 whole = \n"+Theta2)
	
	
// Unroll parameters 
val ary1=Theta1.data.flatten
val ary2=Theta2.data.flatten
println("Flattened Theta2 whole = \n"+ary2.mkString("\n"))
val nn_params=(ary1 ++ ary2).asVector
println("nn_params(30)="+nn_params(30))
//pause()
// val nn_params = ( (Theta1.data.flatten) ++ (Theta2.data.flatten)).asVector @check - this doesn't work

/*
   ================ Part 3: Compute Cost (Feedforward) ================
 
  To the neural network, you should first start by implementing the
  feedforward part of the neural network that returns the cost only. You
  should complete the code in nnCostFunction.m to return cost. After
  implementing the feedforward to compute the cost, you can verify that
  your implementation is correct by verifying that you get the same cost
  as us for the fixed debugging parameters.

  We suggest implementing the feedforward cost *without* regularization
  first so that it will be easier for you to debug. Later, in part 4, you
  will get to implement the regularized cost.
*/
printf("\nFeedforward Using Neural Network ...\n")

// Weight regularization parameter (we set this to 0 here).
val lambda3 = 0;
val (j1,grad)= nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X.toDense, y, lambda3); //@check why is capital J1 not allowed?

printf("Cost at parameters (loaded from ex4weights): %f "+
         "\n(this value should be about 0.287629)\n", j1); 

printf("\nProgram paused. Press enter to continue.\n");
//pause();

/*
  =============== Part 4: Implement Regularization ===============
 
  Once your cost function implementation is correct, you should now
  continue to implement the regularization with the cost.

*/
printf("\nChecking Cost Function (w/ Regularization) ... \n")

// Weight regularization parameter (we set this to 1 here).
val lambda4 = 1;

val (j4,grad4) = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X.toDense, y, lambda4);

printf("Cost at parameters (loaded from ex4weights): %f "+
         "\n(this value should be about 0.383770)\n", j4);

printf("Program paused. Press enter to continue.\n");
//pause();
/*
  ================ Part 5: Sigmoid Gradient  ================

  Before you start implementing the neural network, you will first
  implement the gradient for the sigmoid function. You should complete the
  code in the sigmoidGradient.m file.

*/
printf("\nEvaluating sigmoid gradient...\n")
//import org.tutysara.SigmoidGradientLib._

val g = sigmoidGradient(DenseMatrix((1.0,-0.5,0.0, 0.5,1.0),(1.0,-0.5,0.0, 0.5,1.0)));
printf("Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ");
println( g);
printf("\n\n");

printf("Program paused. Press enter to continue.\n");
//pause();

/*
   ================ Part 6: Initializing Pameters ================
 
  In this part of the exercise, you will be starting to implment a two
  layer neural network that classifies digits. You will start by
  implementing a function to initialize the weights of the neural network
  (randInitializeWeights.m)
*/
printf("\nInitializing Neural Network Parameters ...\n")

val initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
val initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

// Unroll parameters
val initial_nn_params = (initial_Theta1.data ++ initial_Theta2.data).asVector;

/*
   =============== Part 7: Implement Backpropagation ===============

  Once your cost matches up with ours, you should proceed to implement the
  backpropagation algorithm for the neural network. You should add to the
  code you've written in nnCostFunction.m to return the partial
  derivatives of the parameters.

*/
printf("\nChecking Backpropagation... \n");

//  Check gradients by running checkNNGradients
checkNNGradients();

printf("\nProgram paused. Press enter to continue.\n");
//pause();
/*
   =============== Part 8: Implement Regularization ===============

  Once your backpropagation implementation is correct, you should now
  continue to implement the regularization with the cost and gradient.

*/
printf("\nChecking Backpropagation (w/ Regularization) ... \n")

// Check gradients by running checkNNGradients
val lambda8 = 3;
checkNNGradients(lambda8);

// Also output the costFunction debugging values
val(debug_J,_)  = nnCostFunction(nn_params, input_layer_size,
                          hidden_layer_size, num_labels, X.toDense, y, lambda8);

printf("\n\nCost at (fixed) debugging parameters (w/ lambda = 10): %f "+
         "\n(this value should be about 0.576051)\n\n", debug_J);

printf("Program paused. Press enter to continue.\n");
pause();
  }   

}