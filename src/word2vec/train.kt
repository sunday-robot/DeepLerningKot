package word2vec

import common.createShuffledIndices
import common.printArray
import common.toOnehot
import kotlin.math.min

fun main() {
//    val wordVectorSize = 5
    val wordVectorSize = 2
    val batchSize = 3
    val epochCount = 1000

    // テキストデータから、単語のリストを作成する。
    val wordList = createWordsFromTextFile("alices_adventures_in_wonderland.txt")
//    val wordList = stringToWords("You say goodbye and I say hello.")

    // テキストデータからvocabulary(単語とID(０オリジンのただの通し番号)の対応表)と、
    // corpus(テキストデータを、ボキャブラリー内の単語のIDのリストにしたもの)を、作成する。
    val (vocabulary, corpus) = createVocabularyAndCorpus(wordList)

    // 学習用データ(単語と、その手前及び後の単語のセットのセット)
    val targetAndContextList = createTargetAndContextList(corpus, 1)

    // one-hot値のテーブルを作成する
    val oneHotList = mutableListOf<Array<Float>>()
    for (i in 0.until(vocabulary.size))
        oneHotList.add(toOnehot(i, vocabulary.size))

    // word2vecのニューラルネットワークおよびオプティマイザーを生成する
    val network = createSimpleSkipGram(vocabulary.size, wordVectorSize)
    val optimizer = createSimpleSkipGramAndAdamOptimizer(vocabulary.size, wordVectorSize)

    // word2vecのニューラルネットワークの学習
    for (i in 0.until(epochCount)) {    // エポック数分のループ
        println(i)
        val trainDataIndices = createShuffledIndices(targetAndContextList.size) // 学習データのインデックス値をランダムに並べたリスト
        for (j in 0.until(targetAndContextList.size) step batchSize) {
            network.reset() // ネットワークの重み値の微分値の累積値の０クリア
            val bs = min(batchSize, targetAndContextList.size - j)
            for (k in 0.until(bs)) {
                val idx = trainDataIndices[j + k]
                val tc = targetAndContextList[idx]
                val target = oneHotList[tc.target]
                val context = listOf(oneHotList[tc.context[0]], oneHotList[tc.context[1]])
                network.gradient(context, target)   // 重み値の微分値を求め、累積する
            }
            optimizer.update(network)   // ネ重み値の微分値の累積値に従い、重み値を更新する
        }
    }

    // word2vecのニューラルネットワークの第１レイヤーの重み値を単語ベクトルとして取り出す。
    val wordVectorList = network.wordVectorList()

    // 単語ベクトルの値をコンソールに出力する。(Excelに取り込みやすいように、TSV形式で出力する)
    for (i in 0.until(vocabulary.size)) {
        print("${i}\t ${vocabulary.word(i)}")
        for (e in wordVectorList[i])
            print("\t${e}")
        println()
    }
}
