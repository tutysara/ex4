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
import org.tutysara.Util.pause
import org.tutysara.Util.shuffle
import scala.math.sqrt
/*
 DISPLAYDATA Display 2D data in a nice grid
   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
   stored in X in a nice grid. It returns the figure handle h and the 
   displayed array if requested.
 */
object displayData {

	def apply(X:DenseMatrix[Double], width_in:Int=0):DenseMatrix[Double]={
		val example_width= (if(width_in==0) sqrt (X.numCols) else  width_in).ceil.toInt //expand and make it to int
		//Gray Image
		//colormap(gray);

		//Compute rows, cols
		val m=X.numRows
		val n=X.numCols
		//printf("\nm = %d, n= %d",m,n)
		val example_height = (n / example_width)
		//printf("\nexample_height = %d, example_width= %d",example_height,example_width)
		//Compute number of items to display
		val display_rows = sqrt(m).floor.toInt;
		val display_cols = (m / display_rows).ceil.toInt;
		printf("\ndisplay_rows = %d, display_cols= %d",display_rows,display_cols)
		//Between images padding
		val pad = 1;
		//Setup blank display
		val display_array =DenseMatrix.ones[Double](pad + display_rows * (example_height + pad),
				pad + display_cols * (example_width + pad));

		//Copy each example into a patch on the display array
		var curr_ex = 0;
		var run=true
		for (j  <-0 until display_rows;if(run)){
			for (i <- 0 until display_cols;if (run)){
				
				//printf("\nj = %d, i=%d, curr_ex=%d, run = %b",j,i,curr_ex,run)
				//Copy the patch

				//Get the max value of the patch
				val max_val = X(curr_ex, ::).max.abs
				val patch_col=X(curr_ex,::).toList
				val patch=DenseMatrix.tabulate[Double](example_height,example_width)( //divide by max value
						(i,j)=>patch_col((i*example_height)+j)/max_val
				)
				///display_array(pad + (j - 1) * (example_height + pad) :+ ( 0 to example_height), ...
				//             pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
				//				reshape(X(curr_ex, :), example_height, example_width) / max_val;
				//println("patch = "+patch)
				display_array((pad + j * (example_height + pad)) until (pad + j * (example_height + pad)+example_height)  ,
						(pad + i * (example_width + pad)) until (pad + i * (example_width + pad)+example_width)
				)  :=patch

				curr_ex = curr_ex + 1;
				//printf("\nj = %d, i=%d, curr_ex=%d",j,i,curr_ex)
				
				if (curr_ex >=m){ //do until m since the index starts at 0
					run=false; 
				}
			}
			
			if (curr_ex >=m) {
				run=false;
			}

		}
		//Display Image
		//h = imagesc(display_array, [-1 1]);
		
		figure(1)
		image(display_array.t)
		

		//Do not show axis
		//axis image off

		//drawnow;
		display_array //return the created array
	}

	def main(args: Array[String]): Unit = {}

}