package word2vec

import c5backpropergation.LearnableLayer
import java.io.ObjectInputStream
import java.io.ObjectOutputStream

class SimpleSkipGramOptimizer(
    private val inLayerOptimizer: AdamOptimizer,
    private val outLayerOptimizers: Array<AdamOptimizer>
) {
    constructor(ois: ObjectInputStream) : this(
        AdamOptimizer(ois),
        Array<AdamOptimizer>(ois.readInt()) {
            AdamOptimizer(ois)
        }
    )

    fun serialize(oos: ObjectOutputStream) {
        inLayerOptimizer.serialize(oos)
        oos.writeInt(outLayerOptimizers.size)
        outLayerOptimizers.forEach {
            it.serialize(oos)
        }
    }

    fun update(network: SimpleSkipGram) {
        update(inLayerOptimizer, network.inLayer)
        outLayerOptimizers.indices.forEach { update(outLayerOptimizers[it], network.outLayers[it]) }
    }
}

private fun update(optimizer: AdamOptimizer, layer: LearnableLayer) {
    optimizer.update(layer.getAllParameter(), layer.getAllParameterGradient())
}

private fun update(optimizer: AdamOptimizer, layer: LearnableOneHotLayer) {
    optimizer.update(layer.getAllParameter(), layer.getAllParameterGradient())
}
