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

def linearize(in:DenseMatrix[Double]):DenseVectorCol[Double]={
  val res=DenseVectorCol.tabulate[Double](in.numRows*in.numCols)(
      (n)=>{
        //val i:Int=(n/in.numRows).floor.toInt
        //val j:Int=(n%in.numRows)
        //in(i,j)
        
        val j1:Int=(n/in.numRows).floor.toInt
        val i1:Int=(n%in.numRows)
        in(i1,j1)
      }
      )   
  res
}
def maxindex(in:DenseMatrix[Double],axis:Axis=Axis.Horizontal)={
  val res=DenseVectorCol.zeros[Int](in.numRows)
  in.foreachTriple((i,j,value)=>{
    res(i)=if(value >in(i,res(i))) j else res(i)
  }
   )
  res
}
  def main(args: Array[String]): Unit = {
  		val mat1=reshape(Array(0.0,1,2,3,4,5,6,7, 8.0).asVector,3,3)
  		val mat2=reshape(Array(0.0,1,2,3,4,5).asVector,2,3)
  		println(mat1.t)
  		println()
  		println(mat2.t)
  		/*  		  
  	    val vec1=linearize(mat1)  		
  		val vec2=linearize(mat2)
  		println(vec1)
  		println(vec2)
  		 */
  		
  		val X_hc=DenseMatrix(
       (1.000000 ,  0.084147,  -0.027942,  -0.099999),
       (1.000000  , 0.090930,   0.065699,  -0.053657),
       (1.000000  , 0.014112,   0.098936,   0.042017),
       (1.000000  ,-0.075680,   0.041212,   0.099061),
       (1.000000  ,-0.095892,  -0.054402,   0.065029)
       )
       
  		val mat3=DenseMatrix(
  		    (17.0 ,  24.0,    1.0,    8.0,   15.0),
  		    (23.0,    5.0,   7.0,   14.0,   16.0),
  		    (4.0,    6.0,   13.0,   20.0,   22.0),
  		    (10.0,   12.0,   19.0,  21.0,    3.0),
  		    (11.0,   18.0,  25.0,   2.0,    9.0)
  		    )
  		    
  		println("maxindex(mat3) = \n"+maxindex(mat3)+"\n===============\n")
  		println(maxindex(mat1.t.toDense))
  		println()
  		println(maxindex(mat2.t.toDense))
  }
}