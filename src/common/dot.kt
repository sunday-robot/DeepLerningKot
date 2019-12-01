package common

import nn.SubArray

/**
 * 二つのベクトルの内積
 * @param a 一つ目のベクトル
 * @param b 一つ目のベクトル
 * @return ベクトルaとbの内積
 */
fun dot(a: Array<Double>, b: Array<Double>): Double {
    if (a.size != b.size)
        throw RuntimeException()
    var r = 0.0
    for (i in a.indices)
        r += a[i] * b[i]
    return r
}

fun dotVM(a: Array<Double>, m: Array<Array<Double>>): Array<Double> {
    if (a.size != m.size)
        throw RuntimeException()
    return Array(m[0].size) { j ->
        var sum = 0.0
        a.forEachIndexed { i, e ->
            sum += e * m[i][j]
        }
        sum
    }
}

fun dotVMt(a: Array<Double>, m: Array<Array<Double>>): Array<Double> {
    if (a.size != m[0].size)
        throw RuntimeException()
    return Array(m.size) { j ->
        var sum = 0.0
        a.forEachIndexed { i, e ->
            sum += e * m[j][i]
        }
        sum
    }
}

/**
 * 二つのベクトルの内積
 * @param a 一つ目のベクトル
 * @param b 一つ目のベクトル
 * @return ベクトルaとbの内積
 */
fun dot(a: SubArray<Double>, b: SubArray<Double>): Double {
    if (a.size != b.size)
        throw RuntimeException()
    var r = 0.0
    for (i in 0 until a.size)
        r += a[i] * b[i]
    return r
}

fun dot(a: SubArray<Double>, b: Array<Double>) = dot(a, SubArray(b, 0, b.size))

fun dot(a: Array<Double>, b: SubArray<Double>) = dot(SubArray(a, 0, a.size), b)

fun dotMM(a1: Array<Array<Double>>, a2: Array<Array<Double>>): Array<Array<Double>> {
    val r = a1.size
    val c = a2[0].size
    val n = a2.size
    return Array(r) { i ->
        Array(c) { j ->
            var sum = 0.0
            for (k in 0 until n)
                sum += a1[i][k] * a2[k][j]
            sum
        }
    }
}

fun testAa() {
    val x31 = arrayOf(1.0, 2.0, 3.0)

    val x32 = arrayOf(4.0, 5.0, 6.0)

    println(dot(x31, x32))
}

fun testSs() {
    val x31 = arrayOf(1.0, 2.0, 3.0)
    val sx21 = SubArray(x31, 1, 2)

    val x32 = arrayOf(4.0, 5.0, 6.0)
    val sx22 = SubArray(x32, 0, 2)

    println(dot(sx21, sx22))
}

fun testSa() {
    val x31 = arrayOf(1.0, 2.0, 3.0)
    val sx21 = SubArray(x31, 1, 2)

    val x2 = arrayOf(7.0, 8.0)

    println(dot(sx21, x2))
}

fun testAs() {
    val x2 = arrayOf(7.0, 8.0)

    val x32 = arrayOf(4.0, 5.0, 6.0)
    val sx22 = SubArray(x32, 0, 2)

    println(dot(x2, sx22))
}

fun testAm() {
    val v = arrayOf(10.0, 20.0, 30.0, 40.0)
    val m = arrayOf(arrayOf(1.0, 2.0, 3.0), arrayOf(4.0, 5.0, 6.0), arrayOf(7.0, 8.0, 9.0), arrayOf(10.0, 11.0, 12.0))
    val y = dotVM(v, m)

    printArray(v)
    printArray(m)
    printArray(y)
}

fun testMm() {
    val m1 = arrayOf(arrayOf(1.0, 2.0, 3.0), arrayOf(4.0, 5.0, 6.0))
    val m2 = arrayOf(arrayOf(7.0, 8.0, 9.0, 10.0), arrayOf(11.0, 12.0, 13.0, 14.0), arrayOf(15.0, 16.0, 17.0, 18.0))
    val m3 = dotMM(m1, m2)

    printArray(m1)
    printArray(m2)
    printArray(m3)
}

fun main() {
}
