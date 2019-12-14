package word2vec

fun createSimpleSkipGram(
        vocabularySize: Int,    // 語彙の数（単語の種類数）。SkipGramネットワークの入力および出力のベクトルサイズである。
        wordVectorSize: Int,    // SkipGramネットワークの学習の結果として取得したい単語ベクトルのサイズ。SkipGramネットワークの第1レイヤーのニューロンの数である。
        windowSize: Int)
        : SimpleSkipGram {
    val inLayer = createMatMulOneHotLayer(vocabularySize, wordVectorSize,0.01f)
    val outLayers = Array<MatMulLayer>(windowSize * 2) {
        createMatMulLayer(wordVectorSize,vocabularySize, 0.01f)
    }
    return SimpleSkipGram(inLayer, outLayers)
}
