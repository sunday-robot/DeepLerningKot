package common

fun createShuffledIndices(count: Int): Array<Int> {
    val ml = MutableList(count) { i -> i }
    ml.shuffle()
    return ml.toTypedArray()
}

fun main() {
    val a0 = createShuffledIndices(10)
    printArray(a0)
}
