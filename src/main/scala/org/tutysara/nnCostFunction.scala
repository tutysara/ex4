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
import org.tutysara.Util._

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
   // val Theta1=DenseMatrix.tabulate[Double](hidden_layer_size,input_layer_size+1)(
    //			(i,j)=>nn_params(i*(input_layer_size+1)+j)
    //    	)
     val Theta1=reshape(nn_params,hidden_layer_size,input_layer_size+1)
     println("Theta1 = "+Theta1.numRows,Theta1.numCols)
       //println(Theta1)
     val offset_theta1=hidden_layer_size*(input_layer_size+1)
     
    // val Theta2=DenseMatrix.tabulate[Double](num_labels,hidden_layer_size+1)(
    //			(i,j)=>nn_params(offset_theta1+  (i*(hidden_layer_size+1)+j) )
     //   	)
     val nn_params_offsetted=nn_params(offset_theta1 until nn_params.length)
     val Theta2=reshape(nn_params_offsetted,num_labels,hidden_layer_size+1)
     println("Theta2 = "+Theta2.numRows,Theta2.numCols)
    // println(Theta2)
     
   // println("Theta2 patch 1= \n"+Theta2(0 to 9,0 to 7))
	//println("Theta2 patch 2= \n"+Theta2(0 to 9,8 to 15))
	//println("Theta2 patch 3= \n"+Theta2(0 to 9,16 to 23))
	//println("Theta2 patch 3= \n"+Theta2(0 to 9,24 to 25))
	
	
    // Setup some useful variables
     val m = X.numRows
     
    // You need to return the following variables correctly 
    var  J = 0.0;
    var Theta1_grad = DenseMatrix.zeros[Double](Theta1.numRows,Theta1.numCols)
    var Theta2_grad = DenseMatrix.zeros[Double](Theta2.numRows,Theta2.numCols)
    
    
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
  // println("X_new patch = \n"+X_new(0 to 5,0 to 5))
   println("X_new = \n"+X_new)
	//calculate z_2, a_2, z_3, a_3
	//val test=exp(X)
   println("Theta1 = \n"+Theta1)
    val Z_2=Theta1*(X_new.t); //z_2 for all examples with one column per example
     println("Z_2 =\n"+Z_2) //z_2 iw wrong from second row
    val A_2=sigmoid1(Z_2);
   // println("A_2 ="+A_2.numRows,A_2.numCols)
     println("A_2=\n"+A_2)
    val A_2_new=DenseMatrix.vertcat( DenseMatrix.ones[Double](1,A_2.numCols),A_2); //add bias terms
     //println("A_2_new ="+A_2_new.numRows,A_2_new.numCols)
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
//println("J_all = "+J_all)//J_all is wrong

//val sm1=J_all.data.sum//-26701.827321 should be -1438.145826
//val sm2=(sum(J_all).toArray).sum
//printf("sm1=%f,sm2=%f",sm1,sm2)
J = -(1.0/m) * J_all.data.sum; //sum all the data
//println("J = "+J)

//println("grad = "+grad)
	
/*
   Part 2: Implement the backpropagation algorithm to compute the gradients

         Theta1_grad and Theta2_grad. You should return the partial derivatives of
         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
         Theta2_grad, respectively. After implementing Part 2, you can check
         that your implementation is correct by running checkNNGradients

         Note: The vector y passed into the function is a vector of labels
               containing values from 1..K. You need to map this vector into a 
               binary vector of 1's and 0's to be used with the neural network
               cost function.

         Hint: We recommend implementing backpropagation using a for-loop
               over the training examples if you are implementing it for the 
               first time.
*/

	// we have already calculated z2, a2, z3, a3 for the whole set of examples so, use loops only from point 4 for accumulating
	
	//caluclate delta3 for all examples
val Delta3=A_3 - Y; // has 5000 columns, with one column for each example
	//calculate delta2 for all examples
var Delta2=Theta2.t*Delta3;
Delta2=Delta2(1 until Delta2.numRows,::) ;//leave the bias term before mulitplying with g'(z3)
Delta2=Delta2:*sigmoidGradient(Z_2);

	//calculate Capital delta for the examples
var D2=DenseMatrix.zeros[Double](Theta2.numRows,Theta2.numCols)//initialize accumulators
var D1=DenseMatrix.zeros[Double](Theta1.numRows,Theta1.numCols)
	//println("Delta3 = "+Delta3.numRows,Delta3.numCols)
	//println("Delta2 = "+Delta2.numRows,Delta2.numCols)
	//println("A_2 = "+A_2.numRows,A_2.numCols)
	//println("X_new = "+X_new.numRows,X_new.numCols)
	//println("D2 = "+D2.numRows,D2.numCols)
	
println("Delta3= \n"+Delta3)
println("A_2_new = \n"+A_2_new)
println("Delta2 = \n"+Delta2)

for (i <- 0 until m){//using loop for accumulation
	D2+=Delta3(::,i) * A_2_new(::,i).t; //3x5 X 5x6
	D1+=Delta2(::,i)*X_new(i,::); //5x5 X 5x4
  }

D2=D2:/m;
D1=D1:/m;
println("D1 = \n"+D1)
println("D2 = \n"+D2)
Theta1_grad=DenseMatrix.tabulate[Double](D1.numRows,D1.numCols)( // use regularized result after completing regularization
			(i,j)=>D1(i,j)
    )//clone D1 
Theta2_grad=DenseMatrix.tabulate[Double](D2.numRows,D2.numCols)(
			(i,j)=>D2(i,j)
    )//clone D2

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
J+=( lambda/(2*m) ) * (T1.data.sum + T2.data.sum); //regularized cost

	//calculate regularized gradients

val T1_no_bias=DenseMatrix.tabulate[Double](Theta1.numRows,Theta1.numCols)(
			(i,j)=>Theta1(i,j)
    )//clone Theta1
val T2_no_bias=DenseMatrix.tabulate[Double](Theta2.numRows,Theta2.numCols)(
			(i,j)=>Theta2(i,j)
    )//clone Theta2

//removing bias terms	
T1_no_bias(::,0):=DenseVectorCol.zeros[Double](T1_no_bias.numRows) //make first column of T1 zero  
T2_no_bias(::,0):=DenseVectorCol.zeros[Double](T2_no_bias.numRows) //make first column of T2 zero

//Theta1_grad=D1:+ T1_no_bias:*(lambda/m)
//Theta2_grad=D2:+ T2_no_bias:*(lambda/m);


//println("D1 patch = \n"+D1(0 to 2,0 to 2))
//println("D2 patch = \n"+D2(0 to 2,0 to 2))
println("Theta1_gradpatch = \n"+Theta1_grad(0 to 2,0 to 2))
println("Theta2_grad patch = \n"+Theta2_grad(0 to 2,0 to 2))

val grad=(Theta1_grad.data ++ Theta2_grad.data).asVector
(J,grad)

  }
  def main(args: Array[String]): Unit = {}

}