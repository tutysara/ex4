package org.tutysara
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
import scalala.generic.math.Library._
import org.tutysara.Util._
/*
  CHECKNNGRADIENTS Creates a small neural network to check the
backpropagation gradients
   CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
   backpropagation gradients, it will output the analytical gradients
   produced by your backprop code and the numerical gradients (computed
   using computeNumericalGradient). These two gradient computations should
   result in very similar values.

 */
object checkNNGradients {

	def apply(lambda: Double = 0.0) = {
		val input_layer_size = 3;
		val hidden_layer_size = 5;
		val num_labels = 3;
		val m = 5;
		// We generate some 'random' test data
		val Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
		val Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
		// Reusing debugInitializeWeights to generate X
		val X = debugInitializeWeights(m, input_layer_size - 1);
		val y = DenseVector.tabulate[Double](m)(
				(i) => 1 + ((i + 1) % num_labels))

				// Unroll parameters
				val nn_params = (Theta1.data ++ Theta2.data).asVector;

		// Short hand for cost function
		val costFunc = (p: DenseVectorCol[Double]) => nnCostFunction(p, input_layer_size, hidden_layer_size,
				num_labels, X, y, lambda);

		val (cost, grad) = costFunc(nn_params);
		val numgrad = computeNumericalGradient(costFunc, nn_params);

		// Visually examine the two gradient computations.  The two columns
		// you get should be very similar. 
		println(numgrad, grad); //@check - whether zip operation can be used to group identical index values
		val disp_mat = DenseMatrix.zeros[Double](numgrad.length, 2)
		disp_mat(::, 0) := numgrad
		disp_mat(::, 1) := grad
		println(disp_mat)
		printf("The above two columns you get should be very similar.\n" +
		"(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n");

		// Evaluate the norm of the difference between two solutions.  
		// If you have a correct implementation, and assuming you used EPSILON = 0.0001 
		// in computeNumericalGradient.m, then diff below should be less than 1e-9
		val diff = norm(numgrad - grad, 2) / norm(numgrad + grad, 2);

		printf("If your backpropagation implementation is correct, then \n" +
				"the relative difference will be small (less than 1e-9). \n" +
				"\nRelative Difference: %g\n", diff);
	}
	def main(args: Array[String]): Unit = {
		checkNNGradients()
		pause()
		checkNNGradients(3)
	}

}