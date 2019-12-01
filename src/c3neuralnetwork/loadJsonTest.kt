package c3neuralnetwork

import org.json.JSONArray
import org.json.JSONObject
import org.json.JSONTokener
import java.io.FileInputStream

fun main() {
    val stream = FileInputStream("src/c3neuralnetwork/sample_weight.json")
    val jt = JSONTokener(stream)
    while (jt.more()) {
        val t = jt.nextValue()
        //  println(t.javaClass)
        if (t is JSONObject) {
            for (key in t.keys()) {
                val v = t.get(key)
                println(v.javaClass)
                // println("key=$key, value=$v")
            }
        } else if (t is JSONArray) {
            println(t.length())
        }
        // println(t)
    }
}
