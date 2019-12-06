package word2vec

import common.times

/**
 * 重み値を正規分布の乱数で設定したMatMulレイヤーを生成する。
 */
fun createMatMulLayer2(
        inputSize: Int,                 // 入力ベクトルのサイズ
        outputSize: Int,                // 出力ベクトルのサイズ（レイヤー内のニューロンの数）
        weightStandardDeviation: Float) // 初期重み値の標準偏差
        : MatMulLayer2 {
    return MatMulLayer2(
            weightStandardDeviation * np.random.randn(inputSize * outputSize),
            inputSize,
            outputSize)
}
