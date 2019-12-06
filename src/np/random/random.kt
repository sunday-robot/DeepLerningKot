package np.random

import java.util.*

internal val random = Random()

fun reset(seed:Long) = random.setSeed(seed)
