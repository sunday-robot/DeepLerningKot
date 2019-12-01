package common

inline fun <reified T> createShuffledArray(a: Array<T>): Array<T> {
    val list = MutableList<T>(a.size) { i -> a[i] }
    list.shuffle()
    return list.toTypedArray()
}
