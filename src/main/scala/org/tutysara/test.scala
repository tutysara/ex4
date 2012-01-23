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
object disp {

  def main(args: Array[String]): Unit = {
   val X=DenseMatrix(
       (1.000000 ,  0.084147,  -0.027942,  -0.099999),
       (1.000000  , 0.090930,   0.065699,  -0.053657),
       (1.000000  , 0.014112,   0.098936,   0.042017),
       (1.000000  ,-0.075680,   0.041212,   0.099061),
       (1.000000  ,-0.095892,  -0.054402,   0.065029)
       )

  val Theta1 =DenseMatrix(
		  (0.084147 , -0.027942,  -0.099999,  -0.028790),
		  (0.090930 ,  0.065699 , -0.053657,  -0.096140),
		  (0.014112 ,  0.098936  , 0.042017,  -0.075099),
		  (-0.075680 ,  0.041212 ,  0.099061,   0.014988),
		  (-0.095892 , -0.054402 ,  0.065029,   0.091295)
   )
  val res=Theta1*X.t
  println(res)
   /*
    * shoud be
   8.7469e-02   7.6581e-02   7.2650e-02   7.9289e-02   9.0394e-02
   1.0757e-01   9.8537e-02   8.2509e-02   7.4223e-02   8.1297e-02
   2.8773e-02   2.9898e-02   1.6510e-02   9.1676e-04  -2.5446e-03
  -7.6479e-02  -6.6229e-02  -6.4668e-02  -7.3232e-02  -8.4047e-02
  -1.1142e-01  -1.0147e-01  -8.6391e-02  -8.0052e-02  -8.8277e-02

    */
  }

}