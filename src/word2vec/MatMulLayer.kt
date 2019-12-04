package word2vec

import c5backpropergation.LearnableLayer

/**
 * 通常の全結合層(AffinLayer)のレイヤーから、バイアス値を除いたもの
 */
class MatMulLayer(
        parameter: Array<Float>,
        val inputSize: Int,
        val outputSize: Int) : LearnableLayer(parameter) {

    fun weight(i: Int, j: Int) = parameter(i * outputSize + j)

    private fun addWeightGradient(i: Int, j: Int, gradient: Float) {
        addParameterGradient(i * outputSize + j, gradient)
    }

    override fun evaluate(x: Array<Float>): Array<Float> {
        return Array(outputSize) { j ->
            var sum = 0.0f
            for (i in 0 until inputSize)
                sum += weight(i, j) * x[i]
            sum
        }
    }

    override fun differentiate(dY: Array<Float>, x: Array<Float>): Array<Float> {
        val dX = Array(inputSize) { i ->
            var sum = 0f
            for (j in 0 until outputSize)
                sum += weight(i, j) * dY[j]
            sum
        }

        for (i in 0.until(inputSize))
            for (j in 0 until outputSize)
                addWeightGradient(i, j, x[i] * dY[j])

        return dX
    }
}
