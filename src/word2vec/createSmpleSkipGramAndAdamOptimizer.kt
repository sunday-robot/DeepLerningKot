package word2vec

import common.times

fun createSimpleSkipGramAndAdamOptimizer(
        vocabularySize: Int,    // 語彙の数（単語の種類数）。SkipGramネットワークの入力および出力のベクトルサイズである。
        wordVectorSize: Int)    // SkipGramネットワークの学習の結果として取得したい単語ベクトルのサイズ。SkipGramネットワークの第1レイヤーのニューロンの数である。
        : SimpleSkipGram {
    val inLayer = MatMulLayer(
            0.01f * np.random.randn(vocabularySize * wordVectorSize),
            vocabularySize,
            wordVectorSize)
    val outLayer1 = MatMulLayer(
            0.01f * np.random.randn(wordVectorSize * vocabularySize),
            wordVectorSize,
            vocabularySize)
    val outLayer2 = MatMulLayer(
            0.01f * np.random.randn(wordVectorSize * vocabularySize),
            wordVectorSize,
            vocabularySize)
//    val inLayerOptimizer = createAdam()
    return SimpleSkipGram(inLayer, outLayer1, outLayer2)
}
