package c3neuralnetwork

import org.json.JSONArray
import org.json.JSONObject
import org.json.JSONTokener
import java.io.FileInputStream

fun convertFloatArray(ja: JSONArray) = Array(ja.length()) { i -> ja[i] as Float }

fun convertFloatArray2(ja: JSONArray) = Array(ja.length()) { i -> convertFloatArray(ja[i] as JSONArray) }

fun loadSampleWeight(): Array<Pair<Array<Array<Float>>, Array<Float>>> {
    val stream = FileInputStream("src/c3neuralnetwork/sample_weight.json")
    val jt = JSONTokener(stream)
    val o = jt.nextValue() as JSONObject
    val w1 = convertFloatArray2(o["W1"] as JSONArray)
    val b1 = convertFloatArray(o["b1"] as JSONArray)
    val w2 = convertFloatArray2(o["W2"] as JSONArray)
    val b2 = convertFloatArray(o["b2"] as JSONArray)
    val w3 = convertFloatArray2(o["W3"] as JSONArray)
    val b3 = convertFloatArray(o["b3"] as JSONArray)
    stream.close()
    return arrayOf(Pair(w1, b1), Pair(w2, b2), Pair(w3, b3))
}

fun main() {
    val w = loadSampleWeight()
    println(w)
}
