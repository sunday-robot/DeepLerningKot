package word2vec

class AdamArray(
        private val adamArray: Array<AdamOptimizer>) {
    fun update(
            params: Array<Array<Float>>,
            gradient: Array<Array<Float>>) {
        adamArray.forEachIndexed { i, it -> it.update(params[i], gradient[i]) }
    }
}
