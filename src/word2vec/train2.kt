package word2vec

import common.createShuffledIndices
import common.log.log
import kotlin.math.min

fun main() {
//    val wordVectorSize = 20
//    val wordVectorSize = 5
    val wordVectorSize = 2
    val batchSize = 300
//    val batchSize = 30
//    val batchSize = 3
//    val epochCount = 1000
    val epochCount = 0

    // テキストデータから、単語のリストを作成する。
    val wordList = createWordsFromTextFile("alices_adventures_in_wonderland.txt")
//    val wordList = stringToWords("You say goodbye and I say hello.")

    // テキストデータからvocabulary(単語とID(０オリジンのただの通し番号)の対応表)と、
    // corpus(テキストデータを、ボキャブラリー内の単語のIDのリストにしたもの)を、作成する。
    val (vocabulary, corpus) = createVocabularyAndCorpus(wordList)

    println("語彙数 = ${vocabulary.size}")
    println("単語数 = ${corpus.size}")

    // 学習用データ(単語と、その手前及び後の単語のセットのセット)
    val targetAndContextList = createTargetAndContextList(corpus, 1)

    // word2vecのニューラルネットワークおよびオプティマイザーを生成する
    np.random.reset(0L)
    val network = createSimpleSkipGram2(vocabulary.size, wordVectorSize)
    val optimizer = createSimpleSkipGramAndAdamOptimizer2(vocabulary.size, wordVectorSize)

    // word2vecのニューラルネットワークの学習
    for (i in 0.until(epochCount)) {    // エポック数分のループ
        val trainDataIndices = createShuffledIndices(targetAndContextList.size) // 学習データのインデックス値をランダムに並べたリスト
        for (j in 0.until(targetAndContextList.size) step batchSize) {
            network.reset() // ネットワークの重み値の微分値の累積値の０クリア
            val bs = min(batchSize, targetAndContextList.size - j)
            for (k in 0.until(bs)) {
                val idx = trainDataIndices[j + k]
                val tc = targetAndContextList[idx]
                network.gradient(tc.context, tc.target)   // 重み値の微分値を求め、累積する
            }
            optimizer.update(network)   // ネ重み値の微分値の累積値に従い、重み値を更新する
        }
        var loss = 0f
        targetAndContextList.indices.forEach {
            val tc = targetAndContextList[it]
            loss += network.loss(tc.target, tc.context)
        }
        log("${i}: loss = ${loss}")
    }

    // word2vecのニューラルネットワークの第１レイヤーの重み値を単語ベクトルとして取り出す。
    val wordVectorList = network.wordVectorList()

    // 単語ベクトルの値をコンソールに出力する。(Excelに取り込みやすいように、TSV形式で出力する)
    for (i in 0.until(vocabulary.size)) {
        print("${i}\t${vocabulary.word(i)}\t${vocabulary.count(i)}")
        for (e in wordVectorList[i])
            print("\t${e}")
        println()
    }
}
