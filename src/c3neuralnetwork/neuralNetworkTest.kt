package c3neuralnetwork

import nn.LayerStructure
import nn.NetworkStructure
import nn.NeuralNetwork

fun main() {
    val network = createNetwork()
    val input = arrayOf(1.0, 0.5)
//    val output = network.predict(input)
//    println("${s(input)}\n -> ${s(output)}")
}

private fun createNetwork(): NeuralNetwork {
    return NeuralNetwork(
            NetworkStructure(
                    2,
                    arrayOf(
                            LayerStructure(3),
                            LayerStructure(2),
                            LayerStructure(2)
                    )),
            arrayOf(
                    // layer 1
                    0.1f, 0.1f, 0.2f,
                    0.2f, 0.3f, 0.4f,
                    0.3f, 0.5f, 0.6f,

                    // layer 2
                    0.1f, 0.1f, 0.2f, 0.3f,
                    0.2f, 0.4f, 0.5f, 0.6f,

                    // layer 3
                    0.1f, 0.1f, 0.2f,
                    0.2f, 0.3f, 0.4f))
}

private fun s(array: Array<Double>): String {
    val sb = StringBuilder("{")
    for (value in array) {
        sb.append("$value, ")
    }
    sb.append("}")
    return sb.toString()
}
