package c4study

import common.maxValueIndex

/**
 * 正解率を返す
 * @param y NNの結果のリスト
 * @param t 正解のリスト
 */
fun accuracy(y: Array<Array<Double>>, t: Array<Array<Double>>): Double {
    var correctCount = 0
    for (i in y.indices) {
        val yi = maxValueIndex(y[i])
        val ti = maxValueIndex(t[i])
        if (yi == ti)
            correctCount++
    }
    return correctCount.toDouble() / y.size
}

fun main() {
    val t = arrayOf(
            arrayOf(1.0, 2.0, 3.0), // 2
            arrayOf(1.0, 3.0, 2.0), // 1
            arrayOf(2.0, 1.0, 3.0), // 2
            arrayOf(2.0, 3.0, 1.0), // 1
            arrayOf(3.0, 1.0, 2.0), // 0
            arrayOf(3.0, 2.0, 1.0)) // 0
    val y = arrayOf(
            arrayOf(1.0, 3.0, 2.0), // 1
            arrayOf(2.0, 1.0, 3.0), // 2
            arrayOf(2.0, 3.0, 1.0), // 1
            arrayOf(3.0, 1.0, 2.0), // 0
            arrayOf(3.0, 2.0, 1.0), // 0 OK
            arrayOf(1.0, 2.0, 3.0)) // 2
    println("${accuracy(t, t)}")
    println("${accuracy(y, t)}")
}
