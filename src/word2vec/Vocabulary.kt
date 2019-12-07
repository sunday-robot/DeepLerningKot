package word2vec

/**
 * 単語とそのID（0オリジンの通し番号）の対応表
 */
class Vocabulary(
    private val words: List<String>,
    private val ids: HashMap<String, Int>,
    private val counts: List<Int>
) {
    /**
     * 単語の種類数
     */
    val size get() = words.size

    /**
     * 指定されたIDの単語
     */
    fun word(id: Int) = words[id]

    /**
     * 単語のID(0オリジンのただの通し番号)
     */
    fun id(word: String) = ids[word]!!

    /**
     * 指定されたIDの単語の出現数
     */
    fun count(id: Int) = counts[id]

    /**
     * 指定された単語の出現数
     */
    fun count(word: String) = counts[ids[word]!!]
}
