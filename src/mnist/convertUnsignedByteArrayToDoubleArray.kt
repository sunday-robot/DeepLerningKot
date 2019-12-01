package mnist

/**
 * [byteArray]を、0-255から、0.0-1.0に変換する。
 */
fun convertUnsignedByteArrayToFloatArray(byteArray: Array<Byte>, scale: Float = 255.0f) = Array(byteArray.size) { i ->
    (byteArray[i].toInt() and 0xff).toFloat() / scale
}

/**
 * 動作確認用
 */
fun main() {
    val byteArray = arrayOf(0x00, 0x01, 0x7f, 0x80.toByte(), 0xff.toByte())
    val doubleArray = convertUnsignedByteArrayToFloatArray(byteArray, 255.0f)
    println(byteArray)
    println(doubleArray)
}
