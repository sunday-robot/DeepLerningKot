package word2vec

fun createAdamOptimizer(
    parameterCount: Int, // 重み値の個数
    learningRate: Float = 0.001f,
    beta1: Float = 0.9f,
    beta2: Float = 0.999f
) = AdamOptimizer(
    learningRate, beta1, beta2, 0,
    m = MutableList<Float>(parameterCount) { 0f },
    v = MutableList<Float>(parameterCount) { 0f })
