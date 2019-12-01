package c4study

import common.printArray

private const val h = 0.0001

/**
 * @param f 関数(NNの損失率関数）
 * @param x 関数の引数(NNの重み値のリスト)
 * @return 関数fのxでの微分値
 */
fun numericalGradient(
        f: (Array<Double>) -> Double,
        x: Array<Double>) = Array(x.size) { i ->
    val xi = x[i]
    x[i] = xi + h
    val fxh1 = f(x)
    x[i] = xi - h
    val fxh2 = f(x)
    x[i] = xi
    (fxh1 - fxh2) / (2 * h)
}

fun main() {
    test(arrayOf(1.0, 2.0))
    test(arrayOf(1.0, 2.0, 3.0))
    test(arrayOf(1.0, 2.0, 3.0, 4.0))
}

private fun test(x: Array<Double>) {
    printArray(x)
    val y = numericalGradient(::function2, x)
    print("->")
    printArray(y)
}

private fun function2(x: Array<Double>): Double {
    var sum = 0.0
    x.forEach { e -> sum += e }
    return sum
}
