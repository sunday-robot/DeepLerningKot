package mnist

fun loadTrainImages() = loadImages("mnist/train-images-idx3-ubyte.gz")

fun loadTrainImages2() = loadImages2("mnist/train-images-idx3-ubyte.gz")

fun main() {
    val byteImages = loadTrainImages()
    println(byteImages[0])
    val doubleImages = loadTrainImages2()
    println(doubleImages[0])
}
