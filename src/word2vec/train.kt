package word2vec

import common.createShuffledIndices
import common.printArray
import common.toOnehot
import kotlin.math.min

fun main() {
    val wordVectorSize = 5
    val batchSize = 3
    val epochCount = 1000

    val words = createWordsFromTextFile("alices_adventures_in_wonderland.txt")
//    val words = stringToWords("You say goodbye and I say hello.")

    // テキストデータから"ボキャブラリー"(単なる単語集)と、
    // コーパス(テキストデータを、ボキャブラリー内の単語のインデックス値のリストにしたもの)を、作成する。
    val (vocabulary, corpus) = createVocabularyAndCorpus(words)

    // 学習用データ(単語と、その手前及び後の単語のセットのセット)
    val targetAndContextList = createTargetAndContextList(corpus, 1)

    val oneHotList = mutableListOf<Array<Float>>()   // 「one-hot形式のターゲット」のリスト
    for (i in 0.until(vocabulary.size))
        oneHotList.add(toOnehot(i, vocabulary.size))

    val network = createSimpleSkipGram(vocabulary.size, wordVectorSize)
    val optimizer = createSimpleSkipGramAndAdamOptimizer(vocabulary.size, wordVectorSize)

    for (i in 0.until(epochCount)) {
        println(i)
        val trainDataIndices = createShuffledIndices(targetAndContextList.size)
        for (j in 0.until(targetAndContextList.size) step batchSize) {
            network.reset()
            val bs = min(batchSize, targetAndContextList.size - j)
            for (k in 0.until(bs)) {
                val idx = trainDataIndices[j + k]
                val tc = targetAndContextList[idx]
                val target = oneHotList[tc.target]
                val context = listOf(oneHotList[tc.context[0]], oneHotList[tc.context[1]])
                network.gradient(context, target)
            }
            optimizer.update(network)
        }
    }
    val wordVectors = network.wordVectors()
    for (i in 0.until(vocabulary.size)) {
        print("${i}, ${vocabulary.word(i)}, ")
        printArray(wordVectors[i])
    }
}
