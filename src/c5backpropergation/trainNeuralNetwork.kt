package c5backpropergation

import common.StopWatch
import common.createOneHotArray
import common.createShuffledIndices
import common.log.log
import mnist.MnistImage
import mnist.convertUnsignedByteArrayToFloatArray
import kotlin.math.min
import kotlin.random.Random

val rand = Random(0)

fun main() {
    val epochCount = 10000
    val batchSize = 100
    val learningRate = 0.1f

    val trainImages = mnist.loadTrainImages()
    val trainLabels = mnist.loadTrainLabels()
    val testImages = convertToFloatArray(mnist.loadTestImages())
    val testLabels = convertToFloatArray(mnist.loadTestLabels())

    val network = TwoLayerNet()

    val batchCount = (trainImages.size + batchSize - 1) / batchSize
    val optimizer = TwoLayerNetOptimizer()

    val sw = StopWatch()
    for (i in 0.until(epochCount)) {
        sw.start()
        val trainDataIndices = createShuffledIndices(trainImages.size)
        for (j in 0.until(batchCount)) {
            network.reset()
            val batch = selectBatch(trainImages, trainLabels, trainDataIndices, batchSize, j)
            batch.forEach { e ->
                network.gradient(e.first, e.second)
            }
            optimizer.update(network)
//            network.update(learningRate)
//            log("epoch:${i + 1}/$epochCount, batch:${j + 1}/$batchCount")
            sw.lapse()
        }
        sw.stop()
//        printArray(sw.times())

        var lossSum = 0f
        for (i in testImages.indices) {
            lossSum += network.loss(testImages[i], testLabels[i])
        }
        var loss = lossSum / testImages.size.toFloat()
        log("$i:$loss")
    }
}

fun convertToFloatArray(images: Array<MnistImage>) = Array(images.size) { i ->
    convertUnsignedByteArrayToFloatArray(images[i].intensities)
}

fun convertToFloatArray(labels: Array<Byte>) = Array(labels.size) { i ->
    createOneHotArray(labels[i], 10)
}

/**
 *
 */
fun selectBatch(
        images: Array<MnistImage>,
        labels: Array<Byte>,
        indices: Array<Int>,
        batchSize: Int,
        batchIndex: Int):
        Array<Pair<Array<Float>, Array<Float>>> {
    val bs = min(batchSize, indices.size - batchSize)
    return Array<Pair<Array<Float>, Array<Float>>>(bs) { i ->
        val index = indices[batchIndex * batchSize + i]
        Pair(
                convertUnsignedByteArrayToFloatArray(images[index].intensities),
                createOneHotArray(labels[index], 10))
    }
}
