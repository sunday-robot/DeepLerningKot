package word2vec

import common.printArray
import java.text.BreakIterator

fun stringToWords(s: String): List<String> {
    val l = mutableListOf<String>()
    val boundary = BreakIterator.getWordInstance()
    boundary.setText(s)
    var startIndex = boundary.first()
    var endIndex = boundary.next()
    while (endIndex != BreakIterator.DONE) {
        val w = s.substring(startIndex, endIndex).trim().toLowerCase()
        if (w.isNotEmpty())
            l.add(w)
        startIndex = endIndex
        endIndex = boundary.next()
    }
    return l
}

fun main() {
    printArray(stringToWords("You say goodbye, I say hello.").toTypedArray())
    printArray(stringToWords("").toTypedArray())
}
