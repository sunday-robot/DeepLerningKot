package mnist

fun main() {
    saveImages(loadTrainImages(), "mnist/train")
    saveImages(loadTestImages(), "mnist/test")
}

fun saveImages(images: Array<MnistImage>, basePath: String) {
    for (i in images.indices)
        saveAsPng(images[i], String.format("%s_%05d.png", basePath, i))
}
