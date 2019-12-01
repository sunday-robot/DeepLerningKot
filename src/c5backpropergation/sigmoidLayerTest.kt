package c5backpropergation

import common.printArray

fun main() {
    val l = SigmoidLayer()
    val x = arrayOf(0.0, -1.1, 1.2, -2.2, 3.0, 4.0)
    val y = l.forward(x)
    val dy = arrayOf(0.3, 0.1, -0.2, 0.5, 0.4, 0.3)
    val dOut = l.backward(dy)
    printArray(x)
    printArray(y)
    printArray(dy)
    printArray(dOut)
}
