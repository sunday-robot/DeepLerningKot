package word2vec

import common.toOnehot

fun main() {
    val window_size = 1
    val hidden_size = 5
    val batch_size = 3
    val max_epoch = 1000

    val text = "You say goodbye and I say hello."

    // テキストデータから"ボキャブラリー"(単なる単語集)と、
    // コーパス(テキストデータを、ボキャブラリー内の単語のインデックス値のリストにしたもの)を、作成する。
    val (vocabulary, corpus) = createVocabularyAndCorpus(stringToWords(text))

    // 学習用データ(単語と、その手前及び後の単語のセットのセット)
    val targetAndContextList = createTargetAndContextList(corpus, window_size)

    val targetList = mutableListOf<Array<Float>>()   // 「one-hot形式のターゲット」のリスト
    val contextList = mutableListOf<List<Array<Float>>>()   // 「one-hot形式のコンテキスト」のリスト
    targetAndContextList.forEach {
        targetList.add(toOnehot(it.target, vocabulary.size))
        contextList.add(List<Array<Float>>(window_size * 2) { j ->
            toOnehot(it.context[j], vocabulary.size)
        })
    }

//    model = SimpleCBOW(vocab_size, hidden_size)
    val model = createSimpleSkipGram(vocabulary.size, hidden_size)
//    val optimizer = Adam()
//    val trainer = Trainer(model, optimizer)
//
//    trainer.fit(contexts, target, max_epoch, batch_size)
//    trainer.plot()
//
//    val word_vecs = model.word_vecs
//    for word_id, word in id_to_word.items():
//    print(word, word_vecs[word_id])
}
