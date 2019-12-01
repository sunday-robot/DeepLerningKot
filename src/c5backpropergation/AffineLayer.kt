package c5backpropergation

class AffineLayer(
        parameter: Array<Float>,
        private val inputSize: Int,
        private val outputSize: Int) : LearnableLayer(parameter) {
    /**
     * parameter配列中の、バイアスの開始インデックス
     */
    private val biasIndex = inputSize * outputSize

    private fun weight(i: Int, j: Int) = parameter(i * outputSize + j)

    private fun bias(j: Int) = parameter(biasIndex + j)

    private fun addWeightGradient(i: Int, j: Int, gradient: Float) {
        addParameterGradient(i * outputSize + j, gradient)
    }

    private fun addBiasGradient(j: Int, gradient: Float) {
        addParameterGradient(biasIndex + j, gradient)
    }

    override fun evaluate(x: Array<Float>): Array<Float> {
        return Array(outputSize) { j ->
            var sum = 0.0f
            for (i in 0 until inputSize)
                sum += weight(i, j) * x[i] + bias(j)
            sum
        }
    }

    override fun differentiate(dY: Array<Float>, x: Array<Float>): Array<Float> {
        val dX = Array(inputSize) { i ->
            var sum = 0.0f
            for (j in 0 until outputSize)
                sum += weight(i, j) * dY[j]
            sum
        }

        for (i in 0 until inputSize)
            for (j in 0 until outputSize)
                addWeightGradient(i, j, x[i] * dY[j])
        for (j in 0 until outputSize)
            addBiasGradient(j, dY[j])
        return dX
    }
}
