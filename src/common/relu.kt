package common

/**
 * xが負の場合は0、そうでない場合はxを返すという価値がよくわからない関数だが、
 * DeepLearning(NeuralNetwork?)界隈ではよくつかわれるものであるとのこと。
 */
fun relu(x: Double) = Math.max(x, 0.0)

fun relu(x: Float) = Math.max(x, 0.0f)

//fun relu(x: Array<Double>) = Array(x.size) { i -> relu(x[i]) }

fun main() {
    val x = arrayOf(-1.0, 1.0, 2.0)
    printArray(x)

    val y = x.map { e -> relu(e) }
    println(y)

//    val y2 = relu(x)
//    printArray(y2)
}
