package org.tutysara


import scalala.generic.collection.CanMapValues;

import scalala.operators.{UnaryOp,OpType};
import scalala.scalar.Complex
import scalala.library.Library._
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
 SIGMOIDGRADIENT returns the gradient of the sigmoid function
evaluated at z
   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
   evaluated at z. This should work regardless if z is a matrix or a
   vector. In particular, if z is a vector or matrix, you should return
   the gradient for each element.
*/
object SigmoidGradientLib {
	
  
  def sigmoidGradient[V,That](value : V)(implicit exp : CanSigmoidGradient[V,That]) : That =
    sigmoidGradient(value);
  
  def main(args: Array[String]): Unit = {}

}

object sigmoidGradient{
	
  
  def apply(Z:DenseMatrix[Double]):DenseMatrix[Double]={
    DenseMatrix.tabulate[Double](Z.numRows,Z.numCols)(
    		(i,j)=>{
    		  val g=(1.0/(1+exp(-Z(i,j))))
    		  g*(1-g)
    		}
        )
  }

}
/**
 * Operator type for exp(A).
 *
 * @author dramage
 */
trait OpSigmoidGradient extends scalala.operators.OpType;
object OpSigmoidGradient extends OpSigmoidGradient;

/**
 * Constructiond delegate for exp(A).
 *
 * @author dramage
 */
trait CanSigmoidGradient[A,+RV] extends UnaryOp[A,OpSigmoidGradient,RV] {
  def opType = OpSigmoidGradient;
}

object CanSigmoidGradient{
  implicit object OpI extends CanSigmoidGradient[Int,Double] {
    def apply(v : Int) = {
      val g=(1.0/(1+exp(-v)));
      g*(1-g)
    }
  }

  implicit object OpL extends CanSigmoidGradient[Long,Double] {
    def apply(v : Long) = {
      val g=(1.0/(1+exp(-v)));
      g*(1-g)
    }
  }

  implicit object OpF extends CanSigmoidGradient[Float,Double] {
    def apply(v : Float) = {
      val g=(1.0/(1+exp(-v)));
      g*(1-g)
    }
  }

  implicit object OpD extends CanSigmoidGradient[Double,Double] {
     def apply(v : Double) = {
      val g=(1.0/(1+exp(-v)));
      g*(1-g)
    }
  }

  /*implicit object OpC extends CanSigmoid[Complex,Complex] {
    def apply(v: Complex) = Complex(scala.math.cos(v.imag), scala.math.sin(v.imag)) * scala.math.exp(v.real)
  }*/

  class OpMapValues[From,A,B,To](implicit op : CanSigmoidGradient[A,B], map : CanMapValues[From,A,B,To]) extends CanSigmoidGradient[From,To] {
    def apply(v : From) = map.map(v, op.apply(_));
  }

  implicit def opMapValues[From,A,B,To](implicit map : CanMapValues[From,A,B,To], op : CanSigmoidGradient[A,B])
  : CanSigmoidGradient[From,To] = new OpMapValues[From,A,B,To]()(op, map);

  implicit object OpArrayI extends OpMapValues[Array[Int],Int,Double,Array[Double]]()(OpI,CanMapValues.OpArrayID);
  implicit object OpArrayL extends OpMapValues[Array[Long],Long,Double,Array[Double]]()(OpL,CanMapValues.OpArrayLD);
  implicit object OpArrayF extends OpMapValues[Array[Float],Float,Double,Array[Double]]()(OpF,CanMapValues.OpArrayFD);
  implicit object OpArrayD extends OpMapValues[Array[Double],Double,Double,Array[Double]]()(OpD,CanMapValues.OpArrayDD);
  //implicit object OpArrayC extends OpMapValues[Array[Complex],Complex,Complex,Array[Complex]]()(OpC,CanMapValues.OpArrayCC);
}
