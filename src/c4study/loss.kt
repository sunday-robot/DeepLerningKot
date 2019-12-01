package c4study

import common.crossEntropyError
import common.printArray

/**
 * 交差エントロピー誤差のリストを返すもの
 * @param y NNの結果
 * @param t 教師データ
 */
fun loss(y: Array<Float>, t: Array<Float>) = crossEntropyError(y, t)

/**
 * 交差エントロピー誤差のリストを返すもの
 * @param y NNの結果のリスト
 * @param t 教師データのリスト
 */
fun loss(y: Array<Array<Float>>, t: Array<Array<Float>>) = Array(y.size) { i ->
    crossEntropyError(y[i], t[i])
}

fun main() {
    val y = arrayOf(arrayOf(1.0f, 2.0f, 3.0f), arrayOf(4.0f, 5.0f, 6.0f))
    val t = arrayOf(arrayOf(1.1f, 2.2f, 3.3f), arrayOf(4.4f, 5.5f, 6.6f))
    val r = loss(y, t)
    printArray(r)
}