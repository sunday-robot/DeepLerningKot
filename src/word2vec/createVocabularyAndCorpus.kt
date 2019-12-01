package word2vec

fun createVocabularyAndCorpus(words: List<String>): Pair<Vocabulary, List<Int>> {
    val vocabulary = wordsToVocabulary(words)
    val corpus = createCorpus(words, vocabulary)
    return Pair(vocabulary, corpus)
}
