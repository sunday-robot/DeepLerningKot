package mnist

data class MnistImage(
        val width: Int,
        val height: Int,
        val intensities: Array<Byte>) {
    override fun equals(other: Any?): Boolean {
        if (this === other)
            return true
        if (javaClass != other?.javaClass)
            return false

        other as MnistImage

        if (width != other.width)
            return false
        if (height != other.height)
            return false
        if (!intensities.contentEquals(other.intensities))
            return false

        return true
    }

    override fun hashCode(): Int {
        var result = width
        result = 31 * result + height
        result = 31 * result + intensities.contentHashCode()
        return result
    }
}
