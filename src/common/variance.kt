package common

fun variance(a: Array<Float>): Float {
    var sum = 0.0f
    var sum2 = 0.0f
    for (e in a) {
        sum += e
        sum2 += e * e
    }
    return sum2 / a.size - sum * sum / (a.size * a.size)
}

fun main() {
    val a = arrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f)
    val av = average(a)
    val va = variance(a)
    println("$av, $va")
}