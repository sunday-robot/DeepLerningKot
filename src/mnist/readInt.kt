package mnist

import java.io.BufferedInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

fun readInt(i: BufferedInputStream): Int {
    val bytes = ByteArray(4)
    i.read(bytes)
    if (bytes.size != 4) {
        throw IOException("no more bytes.")
    }
    val bb = ByteBuffer.wrap(bytes)
    bb.order(ByteOrder.BIG_ENDIAN)
    return bb.int
}
