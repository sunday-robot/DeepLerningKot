package nn

/**
 * NNの構造を定義するもの。
 */
class NetworkStructure(
        val inputCount: Int,    // 入力値の個数
        val layerStructures: Array<LayerStructure>)

fun main() {
    val ns = NetworkStructure(
            10,
            arrayOf(
                    LayerStructure(10),
                    LayerStructure(20),
                    LayerStructure(30)))
    println(ns)
}
