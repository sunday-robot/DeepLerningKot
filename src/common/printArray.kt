package common

import nn.SubArray

fun printArray(a: SubArray<*>, level: Int = 0) {
    if (a.isEmpty()) {
        println("{}")
    } else if (a[0] is Array<*> || a[0] is SubArray<*>) {
        // 要素もまた配列である場合
        println("{")
        a.forEachIndexed { index, e ->
            printIndent(level + 1)
            print(index)
            print(":")
            printArray(e as Array<*>, level + 1)
        }
        printIndent(level)
        println("}")
    } else {
        // 要素が配列ではない場合
        printSimpleArray(a)
    }
}

fun printArray(a: Array<*>, level: Int = 0) {
    when {
        a.isEmpty() -> println("{}")
        a[0] is Array<*> -> {
            // 要素もまた配列である場合
            println("{")
            a.forEachIndexed { index, e ->
                printIndent(level + 1)
                print(index)
                print(":")
                printArray(e as Array<*>, level + 1)
            }
            printIndent(level)
            println("}")
        }
        else -> // 要素が配列ではない場合
            printSimpleArray(a)
    }
}

private fun printIndent(level: Int) {
    for (i in 0.until(level))
        print("  ")
}

private fun printSimpleArray(a: Array<*>) {
    print("{")
    for (i in (0..a.size - 2))
        print("${a[i]}, ")
    println("${a[a.size - 1]}}")
}

private fun printSimpleArray(a: SubArray<*>) {
    print("{")
    for (i in (0..a.size - 2))
        print("${a[i]}, ")
    println("${a[a.size - 1]}}")
}

fun main() {
    val a0 = arrayOf(1, 2, 3)
    val a1 = arrayOf(4, 5, 6, 7)
    val a2 = arrayOf(8, 9, 10, 11, 12)
    val aa0 = arrayOf(a0)
    val aa1 = arrayOf(a1, a2)
    val aaa0 = arrayOf(aa0, aa1)
    printArray(a0)
    printArray(aa0)
    printArray(aaa0)
    printArray(arrayOf<Int>())
    printArray(arrayOf(1))
    printArray(arrayOf(1, 2))
    printArray(arrayOf(1, 2, 3))
    printArray(arrayOf("a", "bc", "def"))
}
