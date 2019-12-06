package word2vec

import c5backpropergation.LearnableLayer

class SimpleSkipGramOptimizer2(
    private val inLayerOptimizer: AdamOptimizer,
    private val outLayerOptimizer1: AdamOptimizer,
    private val outLayerOptimizer2: AdamOptimizer
) {
    fun update(network: SimpleSkipGram2) {
        update(inLayerOptimizer, network.inLayer)
        update(outLayerOptimizer1, network.outLayer1)
        update(outLayerOptimizer2, network.outLayer2)
    }
}

private fun update(optimizer: AdamOptimizer, layer: LearnableLayer) {
    optimizer.update(layer.getAllParameter(), layer.getAllParameterGradient())
}

private fun update(optimizer: AdamOptimizer, layer: LearnableLayer2) {
    optimizer.update(layer.getAllParameter(), layer.getAllParameterGradient())
}
