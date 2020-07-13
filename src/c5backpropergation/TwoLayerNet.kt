package c5backpropergation

import common.crossEntropyError
import common.softMax
import java.io.ObjectInputStream
import java.io.ObjectOutputStream

class TwoLayerNet(
    private val affineLayer1: AffineLayer,
    private val affineLayer2: AffineLayer
) {
    private val reluLayer = ReluLayer()

    constructor(
        weightStandardDeviation: Float = 0.01f,
        inputSize: Int = 28 * 28,
        hiddenSize: Int = 50,
        outputSize: Int = 10
    ) : this(
        createAffineLayer(inputSize, hiddenSize, weightStandardDeviation),
        createAffineLayer(hiddenSize, outputSize, weightStandardDeviation)
    )

    constructor(ois: ObjectInputStream) : this(
        ois.readObject() as AffineLayer,
        ois.readObject() as AffineLayer
    )

    fun serialize(oos: ObjectOutputStream) {
        affineLayer1.serialize(oos)
        affineLayer2.serialize(oos)
    }

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

    fun layers(): Array<LearnableLayer> {
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
