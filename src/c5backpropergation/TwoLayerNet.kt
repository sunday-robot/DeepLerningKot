package c5backpropergation

import common.crossEntropyError
import common.softMax
import java.io.Serializable

class TwoLayerNet : Serializable {
    private val weightStandardDeviation = 0.01f
    private val inputSize = 28 * 28
    private val hiddenSize = 50
    private val outputSize = 10

    private val affineLayer1 = createAffineLayer(inputSize, hiddenSize, weightStandardDeviation)
    private val reluLayer = ReluLayer()
    private val affineLayer2 = createAffineLayer(hiddenSize, outputSize, weightStandardDeviation)

    /**
     * 個々のバッチの処理前に呼び、内部変数を初期化する。
     */
    fun reset() {
        affineLayer1.reset()
        affineLayer2.reset()
    }

    /**
     * 重み値の傾き(微分値)を計算する。
     */
    fun gradient(x: Array<Float>, t: Array<Float>) {
        var tmp = affineLayer1.forward(x)
        tmp = reluLayer.forward(tmp)
        tmp = affineLayer2.forward(tmp)
        val y = softMax(tmp)

        // ロス値の計算はしなくてもdYは計算可能(それがsoftmaxを使用する理由にもなっているらしい)

        val dy = Array(y.size) { i -> (y[i] - t[i]) }
        tmp = affineLayer2.backward(dy)
        tmp = reluLayer.backward(tmp)
        affineLayer1.backward(tmp)
    }

    fun layers() : Array<LearnableLayer> {
        return arrayOf(affineLayer1, affineLayer2)
    }

//    fun update(learningRate: Float) {
//        affineLayer1.update(learningRate)
//        affineLayer2.update(learningRate)
//    }

    /**
     * 推論
     */
    fun predict(x: Array<Float>): Array<Float> {
        var tmp = affineLayer1.evaluate(x)
        tmp = reluLayer.evaluate(tmp)
        tmp = affineLayer2.evaluate(tmp)
        return softMax(tmp)
    }

    /**
     * ロス値
     */
    fun loss(x: Array<Float>, t: Array<Float>): Float {
        val y = predict(x)
        return crossEntropyError(y, t)
    }
}
