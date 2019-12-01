package word2vec

fun createAdam(row:Int, column:Int) : AdamArray {
    val x = AdamArray(Array<AdamOptimizer>(row) {AdamOptimizer(column)})
    return x
}
