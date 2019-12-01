package c4study

import common.log.log
import mnist.convertUnsignedByteArrayToFloatArray
import nn.NeuralNetwork
import np.random.nextPositiveValue

fun main() {
    val epochCount = 10000
    val batchSize = 100
    val learningRate = 0.1f

    val trainImages = mnist.loadTrainImages()
    val trainLabels = mnist.loadTrainLabels()
    val testImages = mnist.loadTestImages()
    val testLabels = mnist.loadTestLabels()

    val trainSize = trainImages.size

    val network = createTwoLayerNet(784, 50, 10)

    for (i in 0.until(epochCount)) {
        val grads = Array(batchSize) {
            val j = nextPositiveValue() / trainSize
            log(j.toString())
            val oneHotLabel = byteToOneHotArray(trainLabels[j], 10)
            val grad = numericalGradient(
                    network,
                    { predictResult: Array<Float> -> loss(predictResult, oneHotLabel) },
                    convertUnsignedByteArrayToFloatArray(trainImages[j].intensities))
            grad
        }
        for (i in network.weights.indices) {
            var grad = 0.0f
            for (j in grads.indices)
                grad += grads[j][i]
            network.weights[i] -= learningRate * grad / batchSize
        }
        log(i.toString())
    }
}

private const val h = 0.0001f

/**
 * @param loss 損失率関数
 * @param inputs NNの入力
 * @return NNの微分値
 */
fun numericalGradient(
        network: NeuralNetwork,
        loss: (Array<Float>) -> Float,
        inputs: Array<Float>): Array<Float> {
    val h2 = h * 2
    return Array(network.weights.size) { i ->
        if (i % 1000 == 0)
            log(i.toString())
        val tmp = network.weights[i]
        network.weights[i] = tmp + h
        val loss1 = loss(network.forward(inputs))
        network.weights[i] = tmp - h
        val loss2 = loss(network.forward(inputs))
        network.weights[i] = tmp
        (loss1 - loss2) / h2
    }
}

//fun gradient(
//        network: NeuralNetwork,
//        loss: (Array<Double>) -> Double,
//        inputs: Array<Double>): Array<Double> {
//    val v = network.forward(inputs)
//
//}

private fun byteToOneHotArray(b: Byte, arraySize: Int): Array<Float> {
    val array = Array(arraySize) { 0.0f }
    array[b.toInt()] = 1.0f
    return array
}