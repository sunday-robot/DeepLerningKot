package nn

import common.relu
import common.sigmoid

class LayerStructure(
        val neuronCount: Int,
        val activationFunction: (Float) -> Float = ::sigmoid)

fun main() {
    val ls = LayerStructure(10, ::relu)
    println(ls)
}
