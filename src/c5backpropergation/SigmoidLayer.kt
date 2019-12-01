package c5backpropergation

class SigmoidLayer {
    private var out: Array<Double> = emptyArray()

    fun forward(x: Array<Double>): Array<Double> {
        out = Array(x.size) { i ->
            1 / (1 + Math.exp(-x[i]))
        }
        return out
    }

    fun backward(dOut: Array<Double>) = Array(dOut.size) { i ->
        dOut[i] * (1.0 - out[i]) * out[i]
    }
}
