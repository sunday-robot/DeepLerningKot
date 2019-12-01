import common.printArray
import java.io.*

class SerializableClass(
        val floatArray: Array<Float>) : Serializable

fun main() {
    val so = SerializableClass(arrayOf(0f, 1f, 2f))

    val filePath = "a.bin"

    val oos = ObjectOutputStream(FileOutputStream(filePath))
    oos.writeObject(so)
    oos.close()

    val ois = ObjectInputStream(FileInputStream(filePath))
    val dso = ois.readObject() as SerializableClass
    ois.close()

    printArray(dso.floatArray)
}
