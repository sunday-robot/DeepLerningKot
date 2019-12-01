package c3neuralnetwork

import nn.NeuralNetwork

//fun main() {
//    val nn = loadSampleNetwork()
//    val testImages = loadTestImages()
//    val testLabels = loadTestLabels()
//    for (i in testImages.indices) {
//        val image = testImages[i]
//        val dImage = convertUnsignedByteArrayToDoubleArray(image.intensities, 255.0)
//        val forwardResult = nn.predict(dImage)
//        val result = softMax(forwardResult)
//        val guessedAnswer = maxValueIndex(result)
//        val correctAnswer = testLabels[i]
//        println(String.format("guessed : %d(%f), correct answer : %d", guessedAnswer, result[guessedAnswer], correctAnswer))
//    }
//}
//
private fun loadSampleNetwork(): NeuralNetwork {
    val sw = loadSampleWeight()
    return convertSampleWeightToNeuralNetwork(sw)
}
