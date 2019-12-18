package word2vec

import common.softMax

class SimpleSkipGram(
    val inLayer: MatMulOneHotLayer,
    val outLayers: Array<MatMulLayer>
) {
    /**
     * 個々のバッチの処理前に呼び、内部変数を初期化する。
     */
    fun reset() {
        inLayer.reset()
        outLayers.forEach { it.reset() }
    }

    private fun predict(x: Int): List<Array<Float>> {
        val tmp = inLayer.evaluate(x)
        return List<Array<Float>>(outLayers.size) {
            val tmp1 = outLayers[it].evaluate(tmp)
            softMax(tmp1)
        }
    }

    fun loss(x: Int, t: List<Int>): Float {
        val y = predict(x)
        var l = 0f
        y.indices.forEach { l += crossEntropyError(y[it], t[it]) }
        return l;
    }

    fun gradient(t: List<Int>, x: Int) {
        val ilf = inLayer.forward(x)
        val olbSum = Array<Float>(inLayer.outputSize) {0f};
        outLayers.indices.forEach {
            val olf = softMax(outLayers[it].forward(ilf))
            // ロス値の計算はしなくてもdYは計算可能(それがsoftmaxを使用する理由にもなっているらしい)

            olf[t[it]] -= 1f
            val olb = outLayers[it].backward(olf)
            olbSum.indices.forEach { olbSum[it] += olb[it] }
        }
        inLayer.backward(olbSum)
    }

    fun wordVectorList(): Array<Array<Float>> {
        return Array<Array<Float>>(inLayer.inputSize) { i ->
            Array<Float>(inLayer.outputSize) { j ->
                inLayer.weight(i, j)
            }
        }
    }
}
