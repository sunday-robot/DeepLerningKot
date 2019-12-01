import word2vec.Vocabulary
import word2vec.createTargetAndContextList
import word2vec.createVocabularyAndCorpus
import word2vec.stringToWords

private fun fmt(vocabulary:Vocabulary, index:Int) = String.format("${index}:${vocabulary.word(index)}")

fun main() {
    val window_size = 1

    val text = "You say goodbye and I say hello."

    // テキストデータから"ボキャブラリー"(単なる単語集)と、
    // コーパス(テキストデータを、ボキャブラリー内の単語のインデックス値のリストにしたもの)を、作成する。
    val (vocabulary, corpus) = createVocabularyAndCorpus(stringToWords(text))

    // 学習用データ(単語と、その手前及び後の単語のセットのセット)
    val targetAndContextList = createTargetAndContextList(corpus, window_size)

    targetAndContextList.forEach {
        val t = fmt(vocabulary, it.target)
        val c1 = fmt(vocabulary, it.context[0])
        val c2 = fmt(vocabulary, it.context[1])
        println("${t}->${c1},${c2}")
    }
}
