package common

//fun sum(a: Array<Array<Double>>, dimensionIndex: Int): Array<Double> {
//    val r = a[0].clone()
//    for (i in 1 until a.size) {
//        r += a[i]
//    }
//    return r
//}

fun sum(a: Array<Float>): Float {
    var sum = 0.0f
    a.forEach { e -> sum += e }
    return sum
}
