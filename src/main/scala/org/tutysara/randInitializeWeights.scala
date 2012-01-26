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

/*
 RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
 
incoming connections and L_out outgoing connections
   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
   of a layer with L_in incoming connections and L_out outgoing 
   connections. 

   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
   the first row of W handles the "bias" terms
*/


object randInitializeWeights {
  def apply(L_in:Int,L_out:Int):DenseMatrix[Double]={
    
/*
   ====================== YOUR CODE HERE ======================

 Instructions: Initialize W randomly so that we break the symmetry while
               training the neural network.

 Note: The first row of W corresponds to the parameters for the bias units --------- % first row or column

*/

		  // Randomly initialize the weights to small values
		val epsilon_init = 0.12;
		  // You need to return the following variables correctly     
		val W=DenseMatrix.rand(L_out,1+L_in)
		W:*2*epsilon_init-epsilon_init
  }
  def main(args: Array[String]): Unit = {}

}