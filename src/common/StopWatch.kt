package common

import java.util.*

class StopWatch {
    private var startTime: Long = 0
    private val timeLapse = mutableListOf<Long>()

    fun start() {
        startTime = Calendar.getInstance().timeInMillis
        timeLapse.clear()
    }

    fun stop() = lapse()

    fun lapse() {
        timeLapse.add(Calendar.getInstance().timeInMillis - startTime)
    }

    fun times() = Array(timeLapse.count()) { i -> timeLapse[i] }

    fun time() = timeLapse[timeLapse.size - 1]
}
