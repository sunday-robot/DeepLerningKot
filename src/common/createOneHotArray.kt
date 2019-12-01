package common

fun createOneHotArray(index: Int, size: Int) = Array(size) { i ->
    if (i == index)
        1.0f
    else
        0.0f
}

fun createOneHotArray(index: Byte, size: Int) = createOneHotArray(index.toInt(), size)
