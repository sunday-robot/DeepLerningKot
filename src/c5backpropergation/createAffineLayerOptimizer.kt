package c5backpropergation

import word2vec.AdamOptimizer

fun createAffineLayerOptimizer(inputSize: Int, outputSize: Int) = AdamOptimizer((inputSize + 1) * outputSize)
