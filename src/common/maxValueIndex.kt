package common

/**
 * 最大値の要素の添え字を返す。
 */
fun maxValueIndex(a: Array<Double>): Int {
    var mvi = 0
    for (i in 1.until(a.size)) {
        if (a[i] > a[mvi]) {
            mvi = i
        }
    }
    return mvi
}

fun main() {
    val a = arrayOf(1.0, 2.0, 3.0)
    val i = maxValueIndex(a)
    printArray(a)
    println("->$i")
}
