package common

fun average(a: Array<Float>): Float {
    var sum = 0.0f
    for (e in a)
        sum += e
    return sum / a.size
}
