package common

import java.util.*

/**
 * 正規分布の乱数リストを返す。
 */
fun gaussianRandoms(size: Int, random: Random) = Array(size) { random.nextGaussian().toFloat() }

fun main() {
    val r0 = Random(0)
    val a = gaussianRandoms(1000, r0)
    printArray(a)
    println(average(a))
    println(variance(a))
}
