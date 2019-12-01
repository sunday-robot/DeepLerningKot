package common

import kotlin.math.ln

private const val DELTA = 1e-7f

/**
 * 交差エントロピー誤差
 * @param y NNの結果のリスト
 * @param t 教師データのリスト
 */
fun crossEntropyError(y: Array<Float>, t: Array<Float>): Float {
    val dataSize = y.size
    var sum = 0.0f
    for (i in 0 until dataSize)
        sum += t[i] * ln(y[i] + DELTA)
    return -sum / dataSize
}

fun main() {
    test(arrayOf(0.1f, 0.2f, 0.3f), arrayOf(0.1f, 0.2f, 0.3f))
    test(arrayOf(0.1f, 0.2f, 0.3f), arrayOf(0.2f, 0.3f, 0.4f))
    test(arrayOf(0.1f, 0.2f, 0.3f), arrayOf(0.11f, 0.22f, 0.33f))
}

private fun test(y: Array<Float>, t: Array<Float>) {
    val r = crossEntropyError(y, t)
    println("$r")
}
