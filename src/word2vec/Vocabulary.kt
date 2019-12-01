package word2vec

/**
 * 単語とそのID（0オリジンの通し番号）の対応表
 */
class Vocabulary {
    private val words = arrayListOf<String>()
    private val ids = HashMap<String, Int>()

    fun add(word: String) {
        if (ids.containsKey(word))
            return
        val newId = words.size
        words.add(word)
        ids[word] = newId
    }

    fun id(word: String) = ids[word]!!

    fun word(id: Int) = words[id]

    val size get() = words.size
}
