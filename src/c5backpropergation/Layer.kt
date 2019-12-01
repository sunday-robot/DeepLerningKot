package c5backpropergation

abstract class Layer {
    private var x: Array<Float>? = null

    /**
     * 評価(推論)処理
     * 学習のforward()からも呼ばれる。
     */
    abstract fun evaluate(x: Array<Float>): Array<Float>

    /**
     * 学習のforward
     */
    fun forward(x: Array<Float>): Array<Float> {
        this.x = x
        forwardSub()
        return evaluate(x)
    }

    /**
     * 学習のbackward
     */
    fun backward(dY: Array<Float>): Array<Float> {
        val r = differentiate(dY, x!!)
        x = null
        return r;
    }

    /**
     * forward時に、派生クラス側で追加で行う処理
     */
    protected abstract fun forwardSub()

    /**
     * 微分値を計算する。(backward()から呼ばれるもの)
     */
    protected abstract fun differentiate(dY: Array<Float>, x: Array<Float>): Array<Float>
}
