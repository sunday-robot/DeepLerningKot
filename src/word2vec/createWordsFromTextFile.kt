package word2vec

import java.io.File

fun createWordsFromTextFile(filePath: String) : List<String> {
    val list =  mutableListOf<String>()
    val reader = File(filePath).reader()
    reader.forEachLine { l ->
        stringToWords(l).forEach {
            w->list.add(w)
        }
    }
    return list
}

fun main(args: Array<String>) {
    args.forEach { a->
        val words = createWordsFromTextFile(a)
        print(words)
    }
}
