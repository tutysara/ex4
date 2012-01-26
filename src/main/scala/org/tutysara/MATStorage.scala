package scalala.library

import com.jmatio.io.MatFileReader
import com.jmatio.io.MatFileFilter
import com.jmatio.types.MLArray
import java.io.File
import java.io.FileNotFoundException
import java.io.IOException
import com.jmatio.types.MLDouble
import scalala.tensor.dense.DenseVectorCol
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.Tensor
import scalala.tensor.dense.DenseMatrix
import scalala.generic.collection._
import scalala.operators.Implicits._
import com.jmatio.types.MLInt64
import com.jmatio.types.MLInt8
import com.jmatio.types.MLUInt64
import com.jmatio.types.MLUInt8
import com.jmatio.types.MLSparse

trait MATStorage {

	/*
	 * @param varName
	 * Name of the variable in the the .mat file that should be loaded
	 * @param file
	 * Name of the file from which to load the data
	 * @return
	 * Value of the variable in a type compatible for use with Scalala
	 */
	def load(fileName:String,varName:String ):Option[MLArray]={ //@todo - change this to the super type of all Scalala type
	//inferred type
	  /*Option[Array[_ >: Array[Double] with _2 <: Array[_ >: Double with _1 <: AnyVal]]] where type _1 >: Long with Byte <: AnyVal, type _2 >: Array[Long] with 
	 Array[Byte] <: Array[_ >: Long with Byte <: AnyVal] required: Option[Array[Array[Any]]]*/
			println(varName)
			val filter=new MatFileFilter(Array(varName))
			//@todo
			//handle exception while reading file the Scala way
			val reader=new MatFileReader
			//@todo
			//see the difference between java file and scala file and whether can be interchangeably used
			val file=
				try{
					Option( new File(fileName))
				}catch{
				case ex:FileNotFoundException =>println("File Not Found")
				None
				}
				if(!file.isDefined)
					return None
					//if it has reached this point then file should contain value
			val matContent=
				try{
					val res=reader.read(file.get,filter,MATStorage.policy)
					if(res !=null)
						Option(res)
						else
							None
				}catch{
				case ex:IOException =>println("IO Exception ")
				None
				}
				//proceed only when we have a value from the file

			if(!matContent.isDefined)
				return None
				
	
			Some(matContent.get.get(varName))
					
	}

	/*
	 * @param file
	 * Name of the file from which to load the data
	 * @return
	 * Map of variable names and their values (converted to a type that can be used Scalala)
	 */
	def loadAll(file:String)={
		
	}

	/*
	 * synonym for loadAll
	 */
	def load(file:String)={

	}
	/*
	 * load with variable args
	 */
	def load(file:String,varNames:String*):Map[String,Option[MLArray]]={
		val res=varNames.toList.map(varName=>(varName,load(file,varName)))
		res.toMap
	}

	/*
	 * @return
	 * map of variable names and their types converted to Scalala types
	 */
	def variableNames(file:String)={

	}

	/*
	 * @param in
	 * MLArray read from the .mat file
	 * @return
	 * Values of the MLArray converted into a datatype for easy use Scalala
	 
	//Have implicits to convert MLArray into DenseMatrix or DenseVectorRow or DenseVectorCol
	private def convertToScalala(in:MLArray)={

		//@todo
		//check Scala's pattern matching 

		/*  
		 * until something better than this is found
		 * in match {
    case x:MLArray if x.isDouble()
  }*/
		//no need to check type - lets check whether it handles types automatically

		val res=if(in.isDouble()){
			val res=in.asInstanceOf[MLDouble]
			                        val m=res.getM()
			                        val n=res.getN()
			                        val ret=if(m >1&& n==1){//column vector
			                        	val colAry=res.getArray().flatten
			                        	 new DenseVectorCol(colAry)
			                        }else if(m==1 && n>1){//row vector
			                        	val rowAry=res.getArray().flatten
			                        	 new DenseVectorRow(rowAry)
			                        }else{//matrix - may be even 1X1
			                        	val ary=res.getArray()
			                        	ary.asMatrix
			                        }

		}
		else if(in.isInt16()){
									val res=in.asInstanceOf[MLInt64]
									val m=res.getM()
			                        val n=res.getN()
			                        val ret=if(m >1&& n==1){//column vector
			                        	val colAry=res.getArray().flatten
			                        	 new DenseVectorCol(colAry)
			                        }else if(m==1 && n>1){//row vector
			                        	val rowAry=res.getArray().flatten
			                        	 new DenseVectorRow(rowAry)
			                        }else{//matrix - may be even 1X1
			                        	val ary=res.getArray()
			                        	ary.asMatrix
			                        }
		}
		else {
		  DenseMatrix.eye[Int](5)
		}
		res

	}

	def convertToArray(mlAry:MLArray)={
	  if(mlAry.isDouble()){
	    val res=mlAry.asInstanceOf[MLDouble]
	    Some(res.getArray())
	  }else if(mlAry.isInt64()){
	    val res=mlAry.asInstanceOf[MLInt64]
	    Some(res.getArray())
	  }else if(mlAry.isInt8()){
	    val res=mlAry.asInstanceOf[MLInt8]
	    Some(res.getArray())
	  }else if(mlAry.isUint64()){
	    val res=mlAry.asInstanceOf[MLUInt64]
	    Some(res.getArray())
	  }else if (mlAry.isUint8()){
	    val res=mlAry.asInstanceOf[MLUInt8]
	    Some(res.getArray())
	  }else{
	    println("Type "+mlAry.getType()+ "is not supported")
	    None
	  }
	  
	
	  val test=downCast(new MLArray("test",Array(1,2),3,4))
	  
	 
	  
	implicit def MLArrayToArray(in:MLArray):Array[Array[Double]]={
	   if(in.isDouble){
	     val res=mlAry.asInstanceOf[MLDouble]
	     res.getArray()
	   }else{
	     null
	   }
	   
	 }
	    
	implicit def MLArrayToMLDouble(in:MLArray):MLDouble={
	  in match{
	    case d:MLDouble =>d
	  }
	}
	
	implicit def MLArrayToMLInt64(in:MLArray):MLInt64={
	  in match{
	    case i:MLInt64 =>i
	  }
	}
	
	  implicit def MLArrayToMLSparse(in:MLArray):MLSparse={
	  in match{
	    case s:MLSparse =>s
	  }
	}
	
	}
	  def downCast[T <:MLArray](mlAry:T)={	    
		  if(mlAry.isDouble()){
		    mlAry.asInstanceOf[MLDouble]
		  }else if(mlAry.isInt64()){
		    mlAry.asInstanceOf[MLInt64]	   
		  }else if(mlAry.isInt8()){
		   mlAry.asInstanceOf[MLInt8]	   
		  }else if(mlAry.isUint64()){
		    mlAry.asInstanceOf[MLUInt64]	    
		  }else if (mlAry.isUint8()){
		   mlAry.asInstanceOf[MLUInt8]	    
		  }else{
		    println("Type "+mlAry.getType()+ "is not supported")
		    mlAry
		  }
	  }
	  
	 def downCast2(in:MLArray)={
	   
	   //def downCast2(in: com.jmatio.types.MLArray): Array[_ >: Array[Double] with Array[Long] with Array[Byte] : Double with Long with Byte <
	   in match {
	     case d:MLDouble =>d.getArray()
	     case i64:MLInt64 =>i64.getArray()
	     case i8:MLInt8=>i8.getArray()
	     //case s:MLSparse => s
	     case iu64:MLUInt64 =>iu64.getArray()
	     case iu8:MLUInt8 => iu8.getArray()
	   }
	  }
	 */
//@todo
//put all implicits in a common place
implicit def enrichMLDouble(in:MLDouble)={
  new RichMLDouble(in)
}
implicit def enrichMLInt64(in:MLInt64)={
  new RichMLInt64(in)
}
implicit def enrichMLUInt64(in:MLUInt64)={
  new RichMLUInt64(in)
}
implicit def enrichMLInt8(in:MLInt8)={
  new RichMLInt8(in)
}

implicit def enrichMLUInt8(in:MLUInt8)={
  new RichMLUInt8(in)
}

}
object MATStorage extends MATStorage{
	//file reading policy
	val policy=MatFileReader.MEMORY_MAPPED_FILE

	def main(args:Array[String])={
		println(load("y","/home/tutysra/ex4data.mat"))
	}

}

//not needed i guess, all implicits can be moved into a common package or into MATStorage trait
object RichMLDouble{
 /* implicit def enrichMLDouble(in:MLDouble)={
  new RichMLDouble(in)
}*/
}


class RichMLDouble(in:MLDouble){
  def asVector=in.getArray().flatten.asVector
  def asMatrix=in.getArray().asMatrix
}

class RichMLInt64(in:MLInt64){
  def asVector=in.getArray().flatten.asVector
  def asMatrix=in.getArray().asMatrix
}
class RichMLUInt64(in:MLUInt64){
  def asVector=in.getArray().flatten.asVector
  def asMatrix=in.getArray().asMatrix
}
class RichMLInt8(in:MLInt8){
  def ary=in.getArray().map(_.map(_.toInt)) //convert only when needed both asVector and asMatrix will rarely will be called
  def asVector=ary.flatten.asVector
  def asMatrix=ary.asMatrix
}
class RichMLUInt8(in:MLUInt8){
  def ary=in.getArray().map(_.map(_.toInt))//convert only when needed both asVector and asMatrix will rarely will be called
  def asVector=ary.flatten.asVector
  def asMatrix=ary.asMatrix
}
