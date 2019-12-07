package word2vec

/**
 * 単語のリストから、重複のない”ボキャブラリー"を作成する。
 */
fun createVocabulary(words: List<String>): Vocabulary {
    val creator = VocabularyCreator()
    words.forEach { w -> creator.add(w) }
    return creator.createVocabulary()
}

fun main() {
    val words = listOf("abc", "def", "123", "def")
    val vocabulary = createVocabulary(words)
    println(vocabulary)
}
