package nn

import common.printArray

class SubArray<T>(
        private val array: Array<T>,
        private val startIndex: Int,
        val size: Int) {

    operator fun get(index: Int) = array[startIndex + index]

    operator fun set(index: Int, value: T) {
        array[startIndex + index] = value
    }

    fun isEmpty() = size == 0

    inline fun forEachIndexed(action: (index: Int, T) -> Unit) {
        for (i in 0 until size)
            action(i, this[i])
    }
}

fun main() {
    val array = arrayOf(1, 2, 3, 4, 5)
    val subArray = SubArray(array, 1, 2)

    printArray(array)
    printArray(subArray)
}
