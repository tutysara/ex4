/*
 * Distributed as part of Scalala, a linear algebra library.
 *
 * Copyright (C) 2008- Daniel Ramage
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110 USA
 */

package scalala;
package generic;
package math;


import collection.CanMapValues;

import scalala.operators.{UnaryOp,OpType};
import scalala.scalar.Complex
import scalala.library.Library._

object Library{
   /** Take the sigmoid of the given value. */
  def sigmoid[V,That](value : V)(implicit exp : CanSigmoid[V,That]) : That =
    sigmoid(value);
}
/**
 * Operator type for exp(A).
 *
 * @author dramage
 */
trait OpSigmoid extends operators.OpType;
object OpSigmoid extends OpSigmoid;

/**
 * Constructiond delegate for exp(A).
 *
 * @author dramage
 */
trait CanSigmoid[A,+RV] extends UnaryOp[A,OpSigmoid,RV] {
  def opType = OpSigmoid;
}

object CanSigmoid {
  implicit object OpI extends CanSigmoid[Int,Double] {
    def apply(v : Int) = (1.0/(1+exp(-v)));
  }

  implicit object OpL extends CanSigmoid[Long,Double] {
    def apply(v : Long) = (1.0/(1+exp(-v)));
  }

  implicit object OpF extends CanSigmoid[Float,Double] {
    def apply(v : Float) = (1.0/(1+exp(-v)));
  }

  implicit object OpD extends CanSigmoid[Double,Double] {
    def apply(v : Double) = (1.0/(1+exp(-v)));
  }

  /*implicit object OpC extends CanSigmoid[Complex,Complex] {
    def apply(v: Complex) = Complex(scala.math.cos(v.imag), scala.math.sin(v.imag)) * scala.math.exp(v.real)
  }*/

  class OpMapValues[From,A,B,To](implicit op : CanSigmoid[A,B], map : CanMapValues[From,A,B,To]) extends CanSigmoid[From,To] {
    def apply(v : From) = map.map(v, op.apply(_));
  }

  implicit def opMapValues[From,A,B,To](implicit map : CanMapValues[From,A,B,To], op : CanSigmoid[A,B])
  : CanSigmoid[From,To] = new OpMapValues[From,A,B,To]()(op, map);

  implicit object OpArrayI extends OpMapValues[Array[Int],Int,Double,Array[Double]]()(OpI,CanMapValues.OpArrayID);
  implicit object OpArrayL extends OpMapValues[Array[Long],Long,Double,Array[Double]]()(OpL,CanMapValues.OpArrayLD);
  implicit object OpArrayF extends OpMapValues[Array[Float],Float,Double,Array[Double]]()(OpF,CanMapValues.OpArrayFD);
  implicit object OpArrayD extends OpMapValues[Array[Double],Double,Double,Array[Double]]()(OpD,CanMapValues.OpArrayDD);
  //implicit object OpArrayC extends OpMapValues[Array[Complex],Complex,Complex,Array[Complex]]()(OpC,CanMapValues.OpArrayCC);
}

