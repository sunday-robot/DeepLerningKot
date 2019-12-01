package np.random

import common.printArray

/**
 * 平均 0.0、標準偏差 1.0 のガウス (「正規」) 分布の乱数の配列を返す。
 */
fun randn(count: Int) = Array(count) { random.nextGaussian().toFloat() }

/**
 * 平均 0.0、標準偏差 1.0 のガウス (「正規」) 分布の乱数の2次元配列を返す。
 */
fun randn(row: Int, column: Int) = Array(row) { randn(column) }

fun main() {
    val r = randn(3)
    printArray(r)
    val r2 = randn(2, 3)
    printArray(r2)
}
