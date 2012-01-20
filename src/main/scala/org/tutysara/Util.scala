package org.tutysara

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
  def main(args: Array[String]): Unit = {}

}