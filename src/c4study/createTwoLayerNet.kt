package c4study

import common.gaussianRandoms
import nn.LayerStructure
import nn.NetworkStructure
import nn.NeuralNetwork
import java.util.*

/**
 * 4.5.1 2層ニューラルネットワークのクラス
 */
fun createTwoLayerNet(
        inputSize: Int,  // 入力値の数
        hiddenSize: Int, // 隠れ層(第1層)のニューロンの数
        outputSize: Int, // 出力層(第2層)のニューロンの数
        weightInitStd: Double = 0.01
): NeuralNetwork {
    val networkStructure = NetworkStructure(inputSize, arrayOf(
            LayerStructure(hiddenSize),
            LayerStructure(outputSize)))
    // 各レイヤーの重み値の個数
    val firstLayerWeightCount = (inputSize + 1) * hiddenSize
    val secondLayerWeightCount = (hiddenSize + 1) * outputSize

    // 重み値の配列を生成する(バイアスの重み値は0、通常のエッジの重み値は正規分布乱数を設定する)
    val random = Random(0)
    val weights = gaussianRandoms(firstLayerWeightCount + secondLayerWeightCount, random)
    for (i in (0 until hiddenSize))
        weights[(inputSize + 1) * i] = 0.0f
    for (i in (0 until outputSize))
        weights[firstLayerWeightCount + (hiddenSize + 1) * i] = 0.0f

    // NNを生成し、返す
    return NeuralNetwork(networkStructure, weights)
}

fun main() {
    val nn = createTwoLayerNet(28 * 28, 20, 10)
    print(nn)
}
