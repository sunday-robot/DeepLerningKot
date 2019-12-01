package common

import java.util.*

/**
 * 一様分布の乱数リストを返す。
 */
fun randoms(size: Int, random: Random) = Array(size) { random.nextDouble().toFloat() }

fun main() {
    val r0 = Random(0)
    val a = randoms(100, r0)
    printArray(a)
    println(average(a))
}
