package word2vec

import c5backpropergation.LearnableLayer

class SimpleSkipGramOptimizer2(
    private val inLayerOptimizer: AdamOptimizer,
    private val outLayerOptimizers: Array<AdamOptimizer>
) {
    fun update(network: SimpleSkipGram2) {
        update(inLayerOptimizer, network.inLayer)
        outLayerOptimizers.indices.forEach { update(outLayerOptimizers[it], network.outLayers[it]) }
    }
}

private fun update(optimizer: AdamOptimizer, layer: LearnableLayer) {
    optimizer.update(layer.getAllParameter(), layer.getAllParameterGradient())
}

private fun update(optimizer: AdamOptimizer, layer: LearnableLayer2) {
    optimizer.update(layer.getAllParameter(), layer.getAllParameterGradient())
}
