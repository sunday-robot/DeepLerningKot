package word2vec

fun createCorpus(words: List<String>, vocabulary: Vocabulary): List<Int> {
    return List(words.size) { i ->
        vocabulary.id(words[i])
    }
}

fun main(args: Array<String>) {
    args.forEach {
        val words = createWordsFromTextFile(it)
        val vocabulary = wordsToVocabulary(words)
        val corpus = createCorpus(words, vocabulary)
        corpus.forEach {
            println("${it}:${vocabulary.word((it))}")
        }
    }
}
