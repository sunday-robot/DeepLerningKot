package c5backpropergation

import word2vec.createAdamOptimizer

fun createAffineLayerOptimizer(inputSize: Int, outputSize: Int) =
    createAdamOptimizer(parameterCount = (inputSize + 1) * outputSize)
