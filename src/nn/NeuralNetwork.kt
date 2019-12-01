package nn

/**
 * ニューラルネットワーク
 */
class NeuralNetwork(
        private val structure: NetworkStructure,
        val weights: Array<Float>) { // すべてのニューロンのエッジの重みとバイアス値

    /**
     * @param inputs 入力値のリスト
     * @return 出力値のリスト
     */
    fun forward(inputs: Array<Float>): Array<Float> {
        var weightIndex = 0
        var tmp = inputs
        for (layerStructure in structure.layerStructures) { // レイヤーのループ
            val sumArray = Array(layerStructure.neuronCount) {
                var sum = weights[weightIndex++]
                for (j in 0 until tmp.size) {
                    sum += weights[weightIndex++] * tmp[j]
                }
                layerStructure.activationFunction(sum)
            }
        }
        return tmp
    }

    /**
     * 推論
     * @param inputs 入力値のリスト
     * @return 各レイヤーの出力値のリスト
     */
    fun forward2(inputs: Array<Float>): Array<AZ> {
        var weightIndex = 0
        return Array(structure.layerStructures.size) { i ->
            val layerStructure = structure.layerStructures[i]
            val a = Array(layerStructure.neuronCount) {
                var sum = weights[weightIndex++]
                for (j in 0 until inputs.size)
                    sum += weights[weightIndex++] * inputs[j]
                sum
            }
            val z = Array(layerStructure.neuronCount) { i ->
                layerStructure.activationFunction(a[i])
            }
            AZ(a, z)
        }
    }
}
