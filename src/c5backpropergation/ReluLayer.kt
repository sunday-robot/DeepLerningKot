package c5backpropergation

import kotlin.math.max

class ReluLayer : UnlearnableLayer() {
    override fun evaluate(x: Array<Float>): Array<Float> {
        return Array(x.size) { i -> max(x[i], 0f) }
    }

    override fun differentiate(dY: Array<Float>, x: Array<Float>): Array<Float> {
        val dX = Array(dY.size) { i ->
            if (x[i] > 0)
                dY[i]
            else
                0f
        }
        return dX
    }
}
