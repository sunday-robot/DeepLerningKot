package common

import kotlin.math.exp

/**
 * soft max関数
 *
 * 定義は以下のもの
 *
 * exp(value(i)) / sigma(exp(values(*)))
 *
 * @param values double型の値のリスト
 * @return 各要素を、全要素の合計値で割ったもの
 */
fun softMax(values: Array<Float>): Array<Float> {
    val c = values.max()!!  // exp(values[*])でオーバーフローが発生しないようにするための工夫
    var sum = 0.0f
    val values2 = Array(values.size) { i ->
        val e = exp(values[i] - c)
        sum += e
        e
    }
    return Array(values.size) { i -> values2[i] / sum }
}

fun main() {
    // val input = arrayOf(1.0, 2.0, 3.0, -1.0, -2.0)
    val input = arrayOf(0.3f, 2.9f, 4.0f)
    val output = softMax(input)
    printArray(input)
    printArray(output)
}
