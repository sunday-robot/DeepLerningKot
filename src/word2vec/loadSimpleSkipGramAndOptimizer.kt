package word2vec

import java.io.FileInputStream
import java.io.ObjectInputStream

fun loadSimpleSkipGramAndAdamOptimizer(
    simpleSkipGramFilePath: String,
    simpleSkipGramOptimizerFilePath: String
): Pair<SimpleSkipGram, SimpleSkipGramOptimizer> {
    return Pair(
        SimpleSkipGram(ObjectInputStream(FileInputStream(simpleSkipGramFilePath))),
        SimpleSkipGramOptimizer(ObjectInputStream(FileInputStream(simpleSkipGramOptimizerFilePath))))
}
