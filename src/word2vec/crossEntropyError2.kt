package word2vec

import kotlin.math.ln

private const val DELTA = 1e-7f

/**
 * 交差エントロピー誤差
 * @param y NNの結果のリスト
 * @param t 教師データのリスト
 */
fun crossEntropyError2(y: Array<Float>, t: Int): Float {
    return -ln(y[t] + DELTA) / y.size
}

fun main() {
    test(arrayOf(0.1f, 0.2f, 0.3f), 0)
    test(arrayOf(0.1f, 0.2f, 0.3f), 1)
    test(arrayOf(0.1f, 0.2f, 0.3f), 2)
}

private fun test(y: Array<Float>, t: Int) {
    val r = crossEntropyError2(y, t)
    println("$r")
}
