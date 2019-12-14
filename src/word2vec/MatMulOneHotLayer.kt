package word2vec

/**
 * 通常の全結合層(AffinLayer)のレイヤーから、バイアス値を除いたもの
 */
class MatMulOneHotLayer(
    parameter: Array<Float>,
    val inputSize: Int,
    val outputSize: Int
) : LearnableOneHotLayer(parameter) {

    fun weight(i: Int, j: Int) = parameter(i * outputSize + j)

    private fun addWeightGradient(i: Int, j: Int, gradient: Float) {
        addParameterGradient(i * outputSize + j, gradient)
    }

    override fun evaluate(x: Int): Array<Float> {
        return Array(outputSize) { j ->
            weight(x, j)
        }
    }

    override fun differentiate(dY: Array<Float>, x: Int): Array<Float> {
        val dX = Array(inputSize) { i ->
            var sum = 0f
            for (j in 0 until outputSize)
                sum += weight(i, j) * dY[j]
            sum
        }

        for (j in 0 until outputSize)
            addWeightGradient(x, j, dY[j])

        return dX
    }
}
