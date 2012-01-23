package org.tutysara
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
 * COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
and gives us a numerical estimate of the gradient.
   numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
   gradient of the function J around theta. Calling y = J(theta) should
   return the function value at theta.

 Notes: The following code implements numerical gradient checking, and 
        returns the numerical gradient.It sets numgrad(i) to (a numerical 
        approximation of) the partial derivative of J with respect to the 
        i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
        be the (approximately) the partial derivative of J with respect 
        to theta(i).)
 
*/
object computeNumericalGradient {
  def apply(f:DenseVectorCol[Double]=>(Double,DenseVectorCol[Double]),theta:DenseVectorCol[Double]):
  DenseVectorCol[Double]={
    
val numgrad = DenseVectorCol.zeros[Double](theta.length)
val perturb = DenseVectorCol.zeros[Double](theta.length)
val e = 1e-4;
for (p <- 0 until theta.length){
    // Set perturbation vector
    perturb(p) = e;
    val (loss1,_) = f(theta :- perturb);
    val (loss2,_) = f(theta :+ perturb);
    // Compute Numerical Gradient
    numgrad(p) = (loss2 - loss1) / (2*e);
    perturb(p) = 0;
  }
numgrad
  }
  def main(args: Array[String]): Unit = {}

}