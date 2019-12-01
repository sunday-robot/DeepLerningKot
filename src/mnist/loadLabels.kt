package mnist

import java.io.BufferedInputStream
import java.io.FileInputStream
import java.util.zip.GZIPInputStream

fun loadLabels(filePath: String): Array<Byte> {
    val fis = FileInputStream(filePath)
    val gis = GZIPInputStream(fis)
    val bis = BufferedInputStream(gis)
    val fileId = readInt(bis)
    val dataCount = readInt(bis)
    val labels = bis.readBytes(dataCount)
    bis.close()
    return labels.toTypedArray()
}

fun main() {
    val labels = loadLabels("mnist/train-labels-idx1-ubyte.gz")
    println(labels)
}
