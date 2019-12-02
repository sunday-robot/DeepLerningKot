package word2vec

import c5backpropergation.LearnableLayer

class SimpleSkipGramOptimizer(
    private val inLayerOptimizer: AdamOptimizer,
    private val outLayerOptimizer1: AdamOptimizer,
    private val outLayerOptimizer2: AdamOptimizer
) {
    fun update(network: SimpleSkipGram) {
        update(inLayerOptimizer, network.inLayer)
        update(outLayerOptimizer1, network.outLayer1)
        update(outLayerOptimizer2, network.outLayer2)
    }
}

private fun update(optimizer: AdamOptimizer, layer: LearnableLayer) {
    optimizer.update(layer.getAllParameter(), layer.getAllParameterGradient())
}