package word2vec

import common.times

fun createSimpleSkipGramAndAdamOptimizer2(
        vocabularySize: Int,    // 語彙の数（単語の種類数）。SkipGramネットワークの入力および出力のベクトルサイズである。
        wordVectorSize: Int,    // SkipGramネットワークの学習の結果として取得したい単語ベクトルのサイズ。SkipGramネットワークの第1レイヤーのニューロンの数である。
        windowSize: Int)
        : SimpleSkipGramOptimizer2 {
    val inLayer = createMatMulLayerOptimizer(vocabularySize, wordVectorSize)
    val outLayers = Array<AdamOptimizer>(windowSize * 2) {
        createMatMulLayerOptimizer(wordVectorSize, vocabularySize)
    }
    return SimpleSkipGramOptimizer2(inLayer, outLayers)
}

private fun createMatMulLayerOptimizer(inputSize: Int, outputSize: Int) = AdamOptimizer(inputSize * outputSize)
