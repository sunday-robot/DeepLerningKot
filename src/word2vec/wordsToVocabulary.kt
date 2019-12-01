package word2vec

/**
 * 単語のリストから、重複のない”ボキャブラリー"を作成する。
 */
fun wordsToVocabulary(words: List<String>): Vocabulary {
    // Setを使用して、単語の重複を除去する
    val wordsSet = HashSet<String>()
    words.forEach { w -> wordsSet.add(w) }

    // デバッグがしやすいと思われるので、アルファベット順に並べ替えたリストを作る。(本質的には必要のない処理)
    val wordList = wordsSet.toMutableList()
    wordList.sort()

    val vocabulary = Vocabulary()
    wordList.forEach { e -> vocabulary.add(e) }
    return vocabulary
}

fun main() {
    val words = listOf("abc", "def", "123", "def")
    val vocabulary = wordsToVocabulary(words)
    println(vocabulary)
}
