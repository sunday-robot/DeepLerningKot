package np.random

import common.printArray

fun choice(end: Int, count: Int) = Array(count) { nextPositiveValue() % end }

fun main() {
    val randoms = choice(100, 10)
    printArray(randoms)
}
