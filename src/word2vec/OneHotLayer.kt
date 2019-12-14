package word2vec

/**
 * one-hotレイヤー
 * 入力が通常のベクトルではなく、one-hotのインデックスとなっているレイヤー
 */
abstract class OneHotLayer {
    private var x: Int = 0

    /**
     * 評価(推論)処理
     * 学習のforward()からも呼ばれる。
     */
    abstract fun evaluate(x: Int): Array<Float>

    /**
     * 学習のforward
     */
    fun forward(x: Int): Array<Float> {
        this.x = x
        forwardSub()
        return evaluate(x)
    }

    /**
     * 学習のbackward
     */
    fun backward(dY: Array<Float>): Array<Float> {
        val r = differentiate(dY, x)
        return r;
    }

    /**
     * forward時に、派生クラス側で追加で行う処理
     */
    protected abstract fun forwardSub()

    /**
     * 微分値を計算する。(backward()から呼ばれるもの)
     */
    protected abstract fun differentiate(dY: Array<Float>, x: Int): Array<Float>
}
