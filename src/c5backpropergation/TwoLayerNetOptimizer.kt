package c5backpropergation

class TwoLayerNetOptimizer {
    private val inputSize = 28 * 28
    private val hiddenSize = 50
    private val outputSize = 10

    private val affineLayer1Optimizer = createAffineLayerOptimizer(inputSize, hiddenSize)
    private val affineLayer2Optimizer = createAffineLayerOptimizer(hiddenSize, outputSize)

    fun update(network: TwoLayerNet) {
        val layers = network.layers()
        val l1 = layers[0]
        val l2 = layers[1]
        affineLayer1Optimizer.update(l1.getAllParameter(), l1.getAllParameterGradient())
        affineLayer2Optimizer.update(l2.getAllParameter(), l2.getAllParameterGradient())
    }
}
