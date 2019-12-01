package nn

/**
 * @param a 入力値と重み値の内積で、活性化関数の入力である
 * @param z 活性化関数の出力
 */
class AZ(val a: Array<Float>, private val z : Array<Float>)
