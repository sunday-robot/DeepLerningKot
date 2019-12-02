package word2vec

import common.times

fun createSimpleSkipGramAndAdamOptimizer(
        vocabularySize: Int,    // 語彙の数（単語の種類数）。SkipGramネットワークの入力および出力のベクトルサイズである。
        wordVectorSize: Int)    // SkipGramネットワークの学習の結果として取得したい単語ベクトルのサイズ。SkipGramネットワークの第1レイヤーのニューロンの数である。
        : SimpleSkipGramOptimizer {
    val inLayer = createMatMulLayerOptimizer(vocabularySize, wordVectorSize)
    val outLayer1 = createMatMulLayerOptimizer(wordVectorSize, vocabularySize)
    val outLayer2 = createMatMulLayerOptimizer(wordVectorSize, vocabularySize)
    return SimpleSkipGramOptimizer(inLayer, outLayer1, outLayer2)
}

fun createMatMulLayerOptimizer(inputSize: Int, outputSize: Int) = AdamOptimizer(inputSize * outputSize)
