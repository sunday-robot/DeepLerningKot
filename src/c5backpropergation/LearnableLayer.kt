package c5backpropergation

import java.io.ObjectOutputStream

/**
 * 学習を行う(=更新可能なパラメータを持つ)レイヤー
 */
abstract class LearnableLayer(
    private val parameter: Array<Float>
)   // このレイヤーの全てのパラメータ(重み値、バイアス値）
    : Layer() {
    private var batchCount: Int = 0

    /**
     * 各重み値、バイアス値の微分値の合計値
     */
    private var parameterGradientSum: Array<Float>? = null

    /**
     * 重みの微分値などの内部変数を初期化する。
     * (注意)学習モード時、バッチをforwardする前に呼ぶこと。
     */
    fun reset() {
        batchCount = 0
        parameterGradientSum = Array(parameter.size) { 0f }
    }

    override fun forwardSub() {
        batchCount++
    }

    protected fun parameter(i: Int) = parameter[i]

    protected fun addParameterGradient(i: Int, gradient: Float) {
        parameterGradientSum!![i] += gradient
    }

    fun getAllParameter() = parameter

    //    fun getAllParameterGradient() = parameterGradientSum!! / batchCount.toFloat()
    fun getAllParameterGradient() = parameterGradientSum!!

    fun serialize(oos: ObjectOutputStream) {
        oos.writeObject(parameter)
    }
}
