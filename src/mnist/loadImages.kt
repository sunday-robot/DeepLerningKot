package mnist

import java.io.BufferedInputStream
import java.io.FileInputStream
import java.io.IOException
import java.util.zip.GZIPInputStream

fun loadImages(filePath: String): Array<MnistImage> {
    val fis = FileInputStream(filePath)
    val gis = GZIPInputStream(fis)
    val bis = BufferedInputStream(gis)
    val fileId = readInt(bis)
    val dataCount = readInt(bis)
    val height = readInt(bis)
    val width = readInt(bis)
    val size = height * width
    val images = Array(dataCount) {
        val intensities = ByteArray(size)
        val r = bis.read(intensities)
        if (r != size)
            throw IOException()
        MnistImage(width, height, intensities.toTypedArray())
    }
    gis.close()
    return images
}

/**
 * 動作確認用
 */
fun main() {
    val images = loadImages("mnist/train-images-idx3-ubyte.gz")
    println(images)
}
