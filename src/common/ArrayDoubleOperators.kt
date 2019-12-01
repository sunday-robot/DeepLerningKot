package common

// a x b[]

operator fun Float.times(a: Array<Float>) = Array(a.size) { i ->
    this * a[i]
}

// a x b[][]

operator fun Float.times(a: Array<Array<Float>>) = Array(a.size) { i ->
    this * a[i]
}

// a[] x b

operator fun Array<Float>.div(a: Float) = Array(size) { i ->
    this[i] / a
}

// a[] + b[]

operator fun Array<Double>.plus(a: Array<Double>) = Array(size) { i ->
    this[i] + a[i]
}

// a[] - b[]

operator fun Array<Float>.minus(a: Array<Float>) = Array(size) { i ->
    this[i] - a[i]
}

// a[] x b[][]

// a[][] x b

operator fun Array<Array<Float>>.div(a: Float) = Array(size) { i ->
    this[i] / a
}

// a[][] / b[]

operator fun Array<Array<Float>>.div(a: Array<Float>) = Array(size) { i ->
    this[i] / a[i]
}

// a[][] - b[][]

operator fun Array<Array<Float>>.minus(a: Array<Array<Float>>) = Array(size) { i ->
    this[i] - a[i]
}


// a[] += b[]

operator fun Array<Float>.plusAssign(a: Array<Float>) {
    for (i in indices)
        this[i] += a[i]
}

// a[] -= b[]

operator fun Array<Float>.minusAssign(a: Array<Float>) {
    for (i in indices)
        this[i] -= a[i]
}

// a[][] -= b[][]

operator fun Array<Array<Float>>.minusAssign(a: Array<Array<Float>>) {
    for (i in indices)
        for (j in this[i].indices)
            this[i][j] -= a[i][j]
}

fun main() {
    val a = arrayOf(1.0f, 2.0f)
    val b = arrayOf(10.0f, 20.0f)
    printArray(a - b)
    printArray(a / 2.0f)
    val c = 0.1f * a
    printArray(c)
}
