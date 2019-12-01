package word2vec

import c5backpropergation.selectBatch
import common.createOneHotArray
import common.createShuffledIndices
import common.toOnehot
import mnist.MnistImage
import mnist.convertUnsignedByteArrayToFloatArray
import kotlin.math.min

fun main() {
    val window_size = 1
    val hidden_size = 5
    val batchSize = 3
    val epochCount = 1000

    val text = "You say goodbye and I say hello."

    // テキストデータから"ボキャブラリー"(単なる単語集)と、
    // コーパス(テキストデータを、ボキャブラリー内の単語のインデックス値のリストにしたもの)を、作成する。
    val (vocabulary, corpus) = createVocabularyAndCorpus(stringToWords(text))

    // 学習用データ(単語と、その手前及び後の単語のセットのセット)
    val targetAndContextList = createTargetAndContextList(corpus, window_size)

    val oneHotTargetList = mutableListOf<Array<Float>>()   // 「one-hot形式のターゲット」のリスト
    val oneHotContextList = mutableListOf<List<Array<Float>>>()   // 「one-hot形式のコンテキスト」のリスト
    targetAndContextList.forEach {
        oneHotTargetList.add(toOnehot(it.target, vocabulary.size))
        oneHotContextList.add(List<Array<Float>>(window_size * 2) { j ->
            toOnehot(it.context[j], vocabulary.size)
        })
    }

//    model = SimpleCBOW(vocab_size, hidden_size)
    val model = createSimpleSkipGram(vocabulary.size, hidden_size)
    val optimizer = createSimpleSkipGramAndAdamOptimizer(vocabulary.size, hidden_size)

    val batchCount = (targetAndContextList.size + batchSize - 1) / batchSize

    for (i in 0.until(epochCount)) {
        val trainDataIndices = createShuffledIndices(targetAndContextList.size)
        for (j in 0.until(batchCount)) {
            model.reset()
            val batch = selectBatch(trainImages, trainLabels, trainDataIndices, batchSize, j)

            batch.forEach { e ->
                model.gradient(e.first, e.second)
            }
            optimizer.update(network)

        }
    }
//    val trainer = Trainer(model, optimizer)
//
//    trainer.fit(contexts, target, max_epoch, batch_size)
//    trainer.plot()
//
//    val word_vecs = model.word_vecs
//    for word_id, word in id_to_word.items():
//    print(word, word_vecs[word_id])
}

fun selectBatch(
    images: Array<MnistImage>,
    labels: Array<Byte>,
    indices: Array<Int>,
    batchSize: Int,
    batchIndex: Int):
        Array<Pair<Array<Float>, Array<Float>>> {
    val bs = min(batchSize, indices.size - batchSize)
    return Array<Pair<Array<Float>, Array<Float>>>(bs) { i ->
        val index = indices[batchIndex * batchSize + i]
        Pair(
            convertUnsignedByteArrayToFloatArray(images[index].intensities),
            createOneHotArray(labels[index], 10)
        )
    }
}
