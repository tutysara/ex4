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
import scalala.generic.math.Library._

/*
  NNCOSTFUNCTION Implements the neural network cost function for a two layer

neural network which performs classification
   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
   X, y, lambda) computes the cost and gradient of the neural network. The
   parameters for the neural network are "unrolled" into the vector
   nn_params and need to be converted back into the weight matrices. 
 
   The returned parameter grad should be a "unrolled" vector of the
   partial derivatives of the neural network.
 */
object nnCostFunction {
  
  def apply(nn_params:DenseVectorCol[Double],input_layer_size:Int, hidden_layer_size:Int, num_labels:Int,
      		X:DenseMatrix[Double],y:DenseVectorCol[Double],lambda:Double):(Double,DenseVectorCol[Double])={
	  // Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
	  // for our 2 layer neural network
    val Theta1=DenseMatrix.tabulate[Double](hidden_layer_size,input_layer_size+1)(
    			(i,j)=>nn_params(i*(input_layer_size+1)+j)
        	)
     println("Theta1 = "+Theta1.numRows,Theta1.numCols)
     val offset_theta1=hidden_layer_size*(input_layer_size+1)
     
     val Theta2=DenseMatrix.tabulate[Double](num_labels,hidden_layer_size+1)(
    			(i,j)=>nn_params(offset_theta1+  (i*(hidden_layer_size+1)+j) )
        	)
     println("Theta2 = "+Theta2.numRows,Theta2.numCols)
     
   // println("Theta2 patch 1= \n"+Theta2(0 to 9,0 to 7))
	//println("Theta2 patch 2= \n"+Theta2(0 to 9,8 to 15))
	//println("Theta2 patch 3= \n"+Theta2(0 to 9,16 to 23))
	//println("Theta2 patch 3= \n"+Theta2(0 to 9,24 to 25))
	
	
    // Setup some useful variables
     val m = X.numRows
     
    // You need to return the following variables correctly 
    var  J = 0.0;
    val Theta1_grad = DenseMatrix.zeros[Double](Theta1.numRows,Theta1.numCols)
    val Theta2_grad = DenseMatrix.zeros[Double](Theta2.numRows,Theta2.numCols)
    
    
/* 
 ====================== YOUR CODE HERE ======================

 Instructions: You should complete the code by working through the
               following parts.

 Part 1: Feedforward the neural network and return the cost in the
         variable J. After implementing Part 1, you can verify that your
         cost function computation is correct by verifying the cost
         computed in ex4.m

*/
	//println("Theta1 patch = \n"+Theta1(0 to 5,0 to 5))
	//println("Theta2 patch = \n"+Theta2(0 to 5,0 to 5))
   // println("X = "+X.numRows,X.numCols)
   // println("X patch = \n"+X(0 to 5,0 to 5))
    val X_new = DenseMatrix.horzcat( DenseMatrix.ones[Double](X.numRows,1),X); //add bias terms
    println("X_new = "+X_new.numRows,X_new.numCols)
   //println("X_new patch = \n"+X_new(0 to 5,0 to 5))
	//calculate z_2, a_2, z_3, a_3
	//val test=exp(X)
    val Z_2=Theta1*X_new.t; //z_2 for all examples with one column per example
    // println("Z_2 =\n"+Z_2) //z_2 iw wrong from second row
    val A_2=sigmoid1(Z_2);
    println("A_2 ="+A_2.numRows,A_2.numCols)
     println(A_2)
    val A_2_new=DenseMatrix.vertcat( DenseMatrix.ones[Double](1,A_2.numCols),A_2); //add bias terms
     println("A_2_new ="+A_2_new.numRows,A_2_new.numCols)
    val Z_3=Theta2*A_2_new;
    val A_3=sigmoid1(Z_3);

	//calculate Y for all examples

//y_vec=[1:10]; %don't hard code - it fails in backpropogation where they use a smaller data set with smaller number of output labels to check
//val y_vec=(1 to num_labels toArray).asVector
//val Y=y_vec==y(1); //getting the true vector for label y
val Y=DenseMatrix.tabulate[Int](num_labels,m)(//result matrix
		(i,j)=> { //printf("\ni=%d,j=%d,y(j)=%f",i,j,y(j));
					if(y(j)==(i+1)) 1 else 0
				}
		)
println("Y.size = "+Y.numRows, Y.numCols)
//println("Y = "+Y)
	//calculate cost for all classes in an example and all examples
	
//val J_all=Y.*log(A_3)+(ones(size(Y))-Y).*log(ones(size(A_3))-A_3);
val J_all=Y:*log(A_3):+
		(	(DenseMatrix.ones[Int](Y.numRows,Y.numCols):-Y):*
					log( DenseMatrix.ones[Double](A_3.numRows,A_3.numCols):-A_3)
		)
println("J_all = "+J_all)//J_all is wrong

//val sm1=J_all.data.sum//-26701.827321 should be -1438.145826
//val sm2=(sum(J_all).toArray).sum
//printf("sm1=%f,sm2=%f",sm1,sm2)
J = -(1.0/m) * J_all.data.sum; //sum all the data
//println("J = "+J)
val grad=(Theta1_grad.data ++ Theta2_grad.data).asVector
//println("grad = "+grad)

/*
  Part 3: Implement regularization with the cost function and gradients.


         Hint: You can implement this around the code for
               backpropagation. That is, you can compute the gradients for
               the regularization separately and then add them to Theta1_grad
               and Theta2_grad from Part 2.

*/
	//calculate regularized cost function
	
var T1=DenseMatrix.tabulate[Double](Theta1.numRows,Theta1.numCols)(
			(i,j)=>Theta1(i,j)
    )//clone Theta1
var T2=DenseMatrix.tabulate[Double](Theta2.numRows,Theta2.numCols)(
			(i,j)=>Theta2(i,j)
    )//clone Theta2
	//println("T1 patch = \n"+T1(0 to 5,0 to 5))
	//println("T2 patch = \n"+T2(0 to 5,0 to 5))
T1(::,0):=DenseVector.zeros[Double](T1.numRows); //make first column of T1 zero
T2(::,0):=DenseVector.zeros[Double](T2.numRows);  //make first column of T2 zero
	//println("T1 patch = \n"+T1(0 to 5,0 to 5))
	//println("T2 patch = \n"+T2(0 to 5,0 to 5))
T1=T1:^2;
T2=T2:^2;
	//println("T1 patch = \n"+T1(0 to 5,0 to 5))
	//println("T2 patch = \n"+T2(0 to 5,0 to 5))
J+=( lambda/(2*m) ) * (T1.data.sum + T2.data.sum);

(J,grad)

  }
  def main(args: Array[String]): Unit = {}

}