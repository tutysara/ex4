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
import org.tutysara.Util.pause
import scalanlp.optimize._


object fmincg {
  def apply(cf: (DenseVectorCol[Double])=>(Double,DenseVectorCol[Double]),
		  	init_nn_params:DenseVectorCol[Double],max_iteration:Int=400,memory:Int=3):(DenseVectorCol[Double],Double)={
    
    //Initialize Theta
    

    val f = new DiffFunction[DenseVectorCol[Double]] {    		
	   		
             def calculate(init_theta: DenseVectorCol[Double]) = { 	   		 
               cf(init_theta)
             }
          }
    
   val lbfgs = new LBFGS[DenseVectorCol[Double]](maxIter=max_iteration, m=memory) // m is the memory.
   //anywhere between 3 and 7 is fine. The larger m, the more memory is needed.
   val fminunc=lbfgs
   
   val res=fminunc.minimize(f,init_nn_params)
   (res.asCol, cf(res)._1)
  }
  def main(args: Array[String]): Unit = {}

}