package word2vec

import common.crossEntropyError
import common.softMax

class SimpleSkipGram(
    val inLayer: MatMulLayer,
    val outLayer1: MatMulLayer,
    val outLayer2: MatMulLayer
) {
    /**
     * 個々のバッチの処理前に呼び、内部変数を初期化する。
     */
    fun reset() {
        inLayer.reset()
        outLayer1.reset()
        outLayer2.reset()
    }

    fun predict(x: Array<Float>): List<Array<Float>> {
        var tmp = inLayer.evaluate(x)
        val tmp1 = outLayer1.evaluate(tmp)
        val tmp2 = outLayer2.evaluate(tmp)
        val c1 = softMax(tmp1)
        val c2 = softMax(tmp2)
        return listOf(c1, c2)
    }

    fun loss(x: Array<Float>, t: List<Array<Float>>): Float {
        val y = predict(x)
        val l1 = crossEntropyError(y[0], t[0])
        val l2 = crossEntropyError(y[1], t[1])
        return l1 + l2;
    }

    fun gradient(t: List<Array<Float>>, x: Array<Float>) {
        var tmp = inLayer.forward(x)
        var tmp1 = outLayer1.forward(tmp)
        var tmp2 = outLayer2.forward(tmp)
        val y1 = softMax(tmp1)
        val y2 = softMax(tmp2)

        // ロス値の計算はしなくてもdYは計算可能(それがsoftmaxを使用する理由にもなっているらしい)

        val dy1 = Array(y1.size) { i -> y1[i] - t[0][i] }
        val dy2 = Array(y2.size) { i -> y2[i] - t[1][i] }
        tmp1 = outLayer1.backward(dy1)
        tmp2 = outLayer2.backward(dy2)
        tmp = Array(tmp1.size) { i -> tmp1[i] + tmp2[i] }
        inLayer.backward(tmp)
    }

    fun wordVectors(): Array<Array<Float>> {
        return Array<Array<Float>>(inLayer.inputSize) { i ->
            Array<Float>(inLayer.outputSize) { j ->
                inLayer.weight(i, j)
            }
        }
    }
//    fun updateWeight(learningRate: Float) {
//        inLayer.update(learningRate)
//        outLayer1.update(learningRate)
//        outLayer2.update(learningRate)
//    }
}
