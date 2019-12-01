package c4study

fun meanSquaredError(y: Array<Double>, t: Array<Double>): Double {
    var sum = 0.0
    for (i in y.indices) {
        val d = y[i] - t[i]
        sum += d * d
    }
    return sum / 2
}

fun main() {
    test(arrayOf(1.0, 2.0, 3.0), arrayOf(1.0, 2.0, 3.0))
    test(arrayOf(1.0, 2.0, 3.0), arrayOf(2.0, 3.0, 4.0))
    test(arrayOf(1.0, 2.0, 3.0), arrayOf(1.1, 2.2, 3.3))
}

private fun test(y: Array<Double>, t: Array<Double>) {
    val r = meanSquaredError(y, t)
    println("$r")
}
