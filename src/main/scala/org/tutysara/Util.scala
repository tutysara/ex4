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

object Util {

  /*
   * pauses for input
   * has side effect of reading from console
   */
  def pause()={
    try{
      Console.readChar()
    }catch{
      case _ =>println("Ignoring error message")
    }
  }
  
  // Fisher-Yates shuffle, see: http://en.wikipedia.org/wiki/Fisher–Yates_shuffle
  //copied from - http://stackoverflow.com/q/1259223
def shuffle[T](array: Array[T]): Array[T] = {
        val rnd = new java.util.Random
        for (n <- Iterator.range(array.length - 1, 0, -1)) {
                val k = rnd.nextInt(n + 1)
                val t = array(k); array(k) = array(n); array(n) = t
        }
        return array
}

def reshape(in:DenseVectorCol[Double],rows:Int,cols:Int):DenseMatrix[Double]={
	  DenseMatrix.tabulate[Double](rows,cols)(    
      (i,j)=>in((j*(rows)+i))
  )
}
  def main(args: Array[String]): Unit = {
  		println(reshape(Array(0.0,1,2,3,4,5,6,7, 0.8).asVector,3,3))
  }
}