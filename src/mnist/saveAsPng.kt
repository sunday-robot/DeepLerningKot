package mnist

import java.awt.image.BufferedImage
import java.awt.image.BufferedImage.TYPE_BYTE_GRAY
import java.io.File
import javax.imageio.ImageIO

fun saveAsPng(mnistImage: MnistImage, path:String) {
    val bi = BufferedImage(mnistImage.width, mnistImage.height, TYPE_BYTE_GRAY)
    for (y in 0.until(mnistImage.height))
        for (x in 0.until(mnistImage.width)) {
            val intensity = mnistImage.intensities[y * mnistImage.width + x].toInt()
            bi.setRGB(x, y, intensity or (intensity shl 8) or (intensity shl 16))
        }
    ImageIO.write(bi, "PNG", File(path))
}
