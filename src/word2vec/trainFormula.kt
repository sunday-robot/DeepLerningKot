package word2vec

import common.createShuffledIndices
import common.log.log
import common.printArray
import kotlin.math.max
import kotlin.math.min

fun main() {
    val wordVectorSize = 1    // 本家word2vecのデフォルト値は200次元とのこと
    val windowSize = 1    // 本家word2vecのデフォルト値は5とのこと
    val batchSize = 10
    val epochCount = 10000
    val minValue = -1
    val maxValue = 1
    val vocabularySize = -minValue + 1 + maxValue;

    val targetAndContextList = mutableListOf<TargetAndContext>()
    for (context1 in minValue..maxValue)
        for (context2 in max(minValue - context1, minValue)..min(maxValue - context1, maxValue)) {
            val target = context1 + context2
            targetAndContextList.add(
                TargetAndContext(
                    target - minValue, listOf(context1 - minValue, context2 - minValue)
                )
            )
        }

    // word2vecのニューラルネットワークおよびオプティマイザーを生成する
    np.random.reset(1L)
    val network = createSimpleSkipGram(vocabularySize, wordVectorSize, windowSize)
    val optimizer = createSimpleSkipGramAndAdamOptimizer(vocabularySize, wordVectorSize, windowSize)

    // word2vecのニューラルネットワークの学習
    for (i in 0.until(epochCount)) {    // エポック数分のループ
        val trainDataIndices = createShuffledIndices(targetAndContextList.size) // 学習データのインデックス値をランダムに並べたリスト
        for (j in 0.until(targetAndContextList.size) step batchSize) {
//            log("${i}/${epochCount} - ${j}/${targetAndContextList.size}: initializing weight gradients.")
            network.reset() // ネットワークの重み値の微分値の累積値の０クリア
            val bs = min(batchSize, targetAndContextList.size - j)
            for (k in 0.until(bs)) {
                val idx = trainDataIndices[j + k]
                val tc = targetAndContextList[idx]
//                log("${i}/${epochCount} - ${j + k}/${targetAndContextList.size}: calculating weight gradients.")
                network.gradient(tc.context, tc.target)   // 重み値の微分値を求め、累積する
            }
//            log("${i}/${epochCount} - ${j}/${targetAndContextList.size}: optimizing weights.")
            optimizer.update(network)   // 重み値の微分値の累積値に従い、重み値を更新する
        }
        var loss = 0f
        targetAndContextList.indices.forEach {
            val tc = targetAndContextList[it]
            loss += network.loss(tc.target, tc.context)
        }
        log("${i}: loss = ${loss}")
    }

    // (debug)出来上がったNNで、推論を行う。
    for (i in minValue..maxValue) {
        val p = network.predict(i - minValue)
        println("${i}->")
        for (c in p) {
            printArray(c)
        }
        println("")
    }

    // word2vecのニューラルネットワークの第１レイヤーの重み値を単語ベクトルとして取り出す。
    val wordVectorList = network.wordVectorList()

    // 単語ベクトルの値をコンソールに出力する。(Excelに取り込みやすいように、TSV形式で出力する)
    for (word in minValue..maxValue) {
        print("${word}")
        for (e in wordVectorList[word - minValue])
            print("\t${e}")
        println()
    }
}
