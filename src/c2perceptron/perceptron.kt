package c2perceptron

fun calc(w1: Double, w2: Double, bias: Double, x1: Double, x2: Double): Double {
    val tmp = bias + w1 * x1 + w2 * x2
    return if (tmp >= 0)
        1.0
    else
        0.0
}

fun and(x1: Double, x2: Double): Double {
    return calc(0.5, 0.5, -0.9, x1, x2)
}

fun or(x1: Double, x2: Double): Double {
    return calc(0.5, 0.5, -0.5, x1, x2)
}

fun nand(x1: Double, x2: Double): Double {
    return calc(-0.5, -0.5, 0.9, x1, x2)
}

fun xor(x1: Double, x2: Double): Double {
    return and(or(x1, x2), nand(x1, x2))
}

fun main() {
    println(and(0.0, 0.0))
    println(and(0.0, 1.0))
    println(and(1.0, 0.0))
    println(and(1.0, 1.0))
    println("")

    println(or(0.0, 0.0))
    println(or(0.0, 1.0))
    println(or(1.0, 0.0))
    println(or(1.0, 1.0))
    print("")

    println(nand(0.0, 0.0))
    println(nand(0.0, 1.0))
    println(nand(1.0, 0.0))
    println(nand(1.0, 1.0))
    println("")

    println(xor(0.0, 0.0))
    println(xor(0.0, 1.0))
    println(xor(1.0, 0.0))
    println(xor(1.0, 1.0))
    println("")
}

