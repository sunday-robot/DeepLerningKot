package common

fun main() {
    val sw = StopWatch()
    sw.start()
    Thread.sleep(100)
    sw.lapse()
    Thread.sleep(200)
    sw.lapse()
    Thread.sleep(400)
    sw.stop()
    printArray(sw.times())
}
