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

//SIGMOID Compute sigmoid functoon
//   J = SIGMOID(z) computes the sigmoid of z.
object sigmoid1 {
  
  def apply(Z:DenseMatrix[Double]):DenseMatrix[Double]={
   DenseMatrix.tabulate[Double](Z.numRows,Z.numCols)(
       (i,j)=>(1.0/(1+scala.math.exp(-1*Z(i,j))))
       )
     
   }
  
  def main(args: Array[String]): Unit = {}

}