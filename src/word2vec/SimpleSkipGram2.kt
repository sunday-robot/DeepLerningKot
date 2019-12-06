package word2vec

import common.crossEntropyError
import common.softMax

class SimpleSkipGram2(
    val inLayer: MatMulLayer2,
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

    fun predict(x: Int): List<Array<Float>> {
        var tmp = inLayer.evaluate(x)
        val tmp1 = outLayer1.evaluate(tmp)
        val tmp2 = outLayer2.evaluate(tmp)
        val c1 = softMax(tmp1)
        val c2 = softMax(tmp2)
        return listOf(c1, c2)
    }

    fun loss(x: Int, t: List<Int>): Float {
        val y = predict(x)
        val l1 = crossEntropyError2(y[0], t[0])
        val l2 = crossEntropyError2(y[1], t[1])
        return l1 + l2;
    }

    fun gradient(t: List<Int>, x: Int) {
        var tmp = inLayer.forward(x)
        var tmp1 = outLayer1.forward(tmp)
        var tmp2 = outLayer2.forward(tmp)
        val y1 = softMax(tmp1)
        val y2 = softMax(tmp2)

        // ロス値の計算はしなくてもdYは計算可能(それがsoftmaxを使用する理由にもなっているらしい)

        y1[t[0]] -= 1.0f
        y2[t[1]] -= 1.0f
        tmp1 = outLayer1.backward(y1)
        tmp2 = outLayer2.backward(y2)
        tmp = Array(tmp1.size) { i -> tmp1[i] + tmp2[i] }
        inLayer.backward(tmp)
    }

    fun wordVectorList(): Array<Array<Float>> {
        return Array<Array<Float>>(inLayer.inputSize) { i ->
            Array<Float>(inLayer.outputSize) { j ->
                inLayer.weight(i, j)
            }
        }
    }
}
