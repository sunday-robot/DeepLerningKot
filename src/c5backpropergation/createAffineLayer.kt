package c5backpropergation

import common.times
import np.random.randn

fun createAffineLayer(
        inputSize: Int,                 // 入力ベクトルのサイズ
        outputSize: Int,                // 出力ベクトルのサイズ（レイヤー内のニューロンの数）
        weightStandardDeviation: Float) // 初期重み値の標準偏差
        : AffineLayer {
    val parameter = weightStandardDeviation * randn(inputSize * outputSize + outputSize)
    val biasIndex = inputSize * outputSize
    for (i in 0 until outputSize)
        parameter[biasIndex + i] = 0f
    return AffineLayer(parameter, inputSize, outputSize)
}
