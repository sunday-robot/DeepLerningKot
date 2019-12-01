package c5backpropergation

import common.printArray

fun main() {
    val l = ReluLayer()
    val x = arrayOf(0.0f, -1.1f, 1.2f, -2.2f, 3.0f, 4.0f)
    val y = l.forward(x)
    val dy = arrayOf(0.3f, 0.1f, -0.2f, 0.5f, 0.4f, 0.3f)
    val dOut = l.backward(dy)
    print("x: ")
    printArray(x)
    print("y: ")
    printArray(y)
    print("dy: ")
    printArray(dy)
    print("dx: ")
    printArray(dOut)
}
