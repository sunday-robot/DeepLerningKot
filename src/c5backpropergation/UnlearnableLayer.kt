package c5backpropergation

/**
 * 学習を行なわない(=更新可能なパラメータを持たない)レイヤー
 */
abstract class UnlearnableLayer() : Layer() {
    override fun forwardSub() {}
}
