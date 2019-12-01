package word2vec

import kotlin.math.pow
import kotlin.math.sqrt

// http://arxiv.org/abs/1412.6980v8
// ADAMオプティマイザー
class AdamOptimizer(
        paramCount: Int,    // 重み値の個数
        private val learningRate: Float = 0.001f,
        private val beta1: Float = 0.9f,
        private val beta2: Float = 0.999f) {
    private var count: Int = 0
    private val m = MutableList<Float>(paramCount) { 0f }
    private val v = MutableList<Float>(paramCount) { 0f }

    fun update(
            params: Array<Float>,  // 更新対象の重み値のリスト
            grads: Array<Float>) { // 重み値の傾きのリスト

        count += 1
        val lr = learningRate * sqrt(1f - beta2.pow(count)) / (1f - beta1.pow(count))

        params.indices.forEach { i ->
            m[i] += (1f - beta1) * (grads[i] - m[i])
            v[i] += (1f - beta2) * (grads[i].pow(2) - v[i])

            params[i] -= lr * m[i] / (sqrt(v[i]) + 1e-7f)
        }
    }
}
