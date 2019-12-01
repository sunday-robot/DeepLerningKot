package common.log

import java.util.*

fun log(s: String) {
    val c = Calendar.getInstance()
    val y = c.get(Calendar.YEAR)
    val mon = c.get(Calendar.MONTH)
    val d = c.get(Calendar.DAY_OF_MONTH)
    val h = c.get(Calendar.HOUR_OF_DAY)
    val min = c.get(Calendar.MINUTE)
    val sec = c.get(Calendar.SECOND)
    val ms = c.get(Calendar.MILLISECOND)
    val sts = Throwable().stackTrace
    val st = sts[1]
    val className = st.className
    val methodName = st.methodName
    val lineNumber = st.lineNumber
    println(String.format("%04d/%02d/%02d,%02d:%02d:%02d.%03d, %s#%s(%d), %s",
            y, mon + 1, d, h, min, sec, ms,
            className, methodName, lineNumber,
            s))
}

fun main() {
    log("abc")
}
