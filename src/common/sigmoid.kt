package common

import kotlin.math.exp

fun sigmoid(x: Double) = 1 / (1 + exp(-x))

fun sigmoid(x: Float) = 1 / (1 + exp(-x))

//fun sigmoid(x: Array<Double>) = Array(x.size) { i -> sigmoid(x[i]) }

fun main() {
    val x = listOf(-1.0, 1.0, 2.0)
    val y = x.map { e -> sigmoid(e) }
    println(x)
    println(y)
}
