package word2vec

class VocabularyCreator {
    private val wordCounts = HashMap<String, Int>()

    /**
     * 単語を追加する
     */
    fun add(word: String) {
        if (wordCounts.containsKey(word))
            wordCounts[word] = wordCounts[word]!! + 1
        else
            wordCounts[word] = 1
    }

    /**
     * Vocabularyを生成する
     */
    fun createVocabulary(): Vocabulary {
        // デバッグなどがしやすいと思われるので、IDはアルファベット順に割り当てる。(本質的には必要のない処理)
        val words = wordCounts.keys.toMutableList()
        words.sort()
        val ids = HashMap<String, Int>()
        words.forEachIndexed() {id, word->
            ids[word] = id
        }
        val counts = List<Int>(words.size) {
            wordCounts[words[it]]!!
        }
        return Vocabulary(words, ids, counts)
    }
}
