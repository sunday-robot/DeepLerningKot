package common

fun toOnehot(id: Int, length: Int) = Array(length) { i ->
    if (i == id) 1.0f else 0.0f
}

fun main() {
    printArray(toOnehot(0, 2))
    printArray(toOnehot(1, 3))
}
