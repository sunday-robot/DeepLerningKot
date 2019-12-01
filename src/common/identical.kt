package common

fun identical(x: Double) = x

fun identical(x: Array<Double>) = x

fun main() {
    println(identical(1.0))
    printArray(identical(arrayOf(1.0, 2.0)))
}
