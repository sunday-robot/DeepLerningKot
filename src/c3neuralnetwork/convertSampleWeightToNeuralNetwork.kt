package c3neuralnetwork

import nn.LayerStructure
import nn.NetworkStructure
import nn.NeuralNetwork

fun convertSampleWeightToNeuralNetwork(sampleWeights: Array<Pair<Array<Array<Float>>, Array<Float>>>): NeuralNetwork {
    // まずはweightsを走査し、重み値の総数をカウントし、NNの全ての重み値を格納する配列を確保する
    val inputCount = sampleWeights[0].first.size
    val layerStructures = mutableListOf<LayerStructure>()
    var weightCount = 0 // エッジの重みとバイアス値の数
    var previousNeuronCount = inputCount
    sampleWeights.forEach { layer ->
        val edgeWeight = layer.first
        val neuronCount = edgeWeight[0].size
        layerStructures.add(LayerStructure(neuronCount))
        weightCount += neuronCount * (previousNeuronCount + 1)   // "+1"はバイアスの数
        previousNeuronCount = neuronCount
    }
    val networkStructure = NetworkStructure(inputCount, layerStructures.toTypedArray())

    // 1次元の重み値配列を作る。
    var layerIndex = 0
    var neuronIndex = 0
    var edgeIndex = -1
    val weights = Array(weightCount) {
        val w: Float
        if (edgeIndex == -1) {
            // バイアス値
            w = sampleWeights[layerIndex].second[neuronIndex]
            edgeIndex = 0
        } else {
            // エッジの重み
            w = sampleWeights[layerIndex].first[edgeIndex][neuronIndex]
            if (edgeIndex == sampleWeights[layerIndex].first.size - 1) {
                edgeIndex = -1
                if (neuronIndex == sampleWeights[layerIndex].second.size - 1) {
                    neuronIndex = 0
                    layerIndex++
                } else
                    neuronIndex++
            } else
                edgeIndex++
        }
        w
    }

    return NeuralNetwork(networkStructure, weights)
}
