package np.random

import common.printArray

/**
 * 一様分布の乱数の配列を返す。
 */
fun rand(count: Int) = Array(count) {
    random.nextDouble()
}

/**
 * 一様分布の乱数の2次元配列を返す。
 */
fun rand(row: Int, column: Int) = Array(row) {
    rand(column)
}

fun main() {
    val r = rand(3)
    printArray(r)
    val r2 = rand(2, 3)
    printArray(r2)
}
