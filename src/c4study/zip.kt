package c4study

import common.printArray

/**
 * 二つの任意の配列と、これら配列の要素を引数としてとる関数を引数として
 */
inline fun <T1, T2, reified T3> zip(a1: Array<T1>, a2: Array<T2>, f: (T1, T2) -> T3): Array<T3> = Array(a1.size) { i ->
    f(a1[i], a2[i])
}

fun main() {
    val a1 = arrayOf(1, 2, 3)
    val a2 = arrayOf("0.1", "0.2", "0.3")
    val a3 = zip(a1, a2, ::f)
    printArray(a1)
    printArray(a2)
    printArray(a3)
}

fun f(s1: Int, s2: String): Double {
    return s1.toDouble() + s2.toDouble()
}
