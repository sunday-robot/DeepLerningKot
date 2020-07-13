package word2vec

import java.io.ObjectInputStream
import java.io.ObjectOutputStream
import kotlin.math.pow
import kotlin.math.sqrt

// http://arxiv.org/abs/1412.6980v8
// ADAMオプティマイザー
class AdamOptimizer(
    private val learningRate: Float,
    private val beta1: Float,
    private val beta2: Float,
    private var count: Int,
    private val m: MutableList<Float>,
    private val v: MutableList<Float>
) {
    constructor(ois: ObjectInputStream) : this(
        ois.readFloat(),
        ois.readFloat(),
        ois.readFloat(),
        ois.readInt(),
        ois.readObject() as MutableList<Float>,
        ois.readObject() as MutableList<Float>
    )

    fun serialize(oos: ObjectOutputStream) {
        oos.writeFloat(learningRate)
        oos.writeFloat(beta1)
        oos.writeFloat(beta2)
        oos.writeInt(count)
        oos.writeObject(m)
        oos.writeObject(v)
    }

    fun update(
        params: Array<Float>,   // 更新対象の重み値のリスト
        grads: Array<Float>     // 重み値の傾きのリスト
    ) {
        count += 1
        val lr = learningRate * sqrt(1f - beta2.pow(count)) / (1f - beta1.pow(count))

        params.indices.forEach { i ->
            m[i] += (1f - beta1) * (grads[i] - m[i])
            v[i] += (1f - beta2) * (grads[i].pow(2) - v[i])

            params[i] -= lr * m[i] / (sqrt(v[i]) + 1e-7f)
        }
    }
}
