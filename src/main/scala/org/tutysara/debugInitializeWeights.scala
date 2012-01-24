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
import scala.math.sin
/*
  DEBUGINITIALIZEWEIGHTS Initialize the weights of a layer with fan_in
 
incoming connections and fan_out outgoing connections using a fixed
strategy, this will help you later in debugging
   W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights 
   of a layer with fan_in incoming connections and fan_out outgoing 
   connections using a fix set of values

   Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
   the first row of W handles the "bias" terms
*/
object debugInitializeWeights {
  def apply(fan_out:Int, fan_in:Int):DenseMatrix[Double]={
    DenseMatrix.tabulate[Double](fan_out,fan_in+1)(
     // (i,j)=>sin((i*(fan_in+1)+j+1))
      (i,j)=>sin((j*(fan_out)+i+1))/10
  )
  }
  def main(args: Array[String]): Unit = {
    println(debugInitializeWeights(3,3))
  }

}