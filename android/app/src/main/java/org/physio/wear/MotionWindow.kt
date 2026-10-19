package org.physio.wear

import kotlin.math.abs
import kotlin.math.sqrt

/** Including gravity, g units. Shared contract: Python physio.processing.acceleration. */
object MotionFeatures {
    const val BIN_NANOS = 62_500_000L
    const val WINDOW_BINS = 960
    const val GRAVITY = 9.80665

    fun extract(values: DoubleArray): FloatArray {
        require(values.size == WINDOW_BINS && values.all { it.isFinite() })
        val mean = values.average()
        val variance = values.sumOf { (it - mean) * (it - mean) } / values.size
        var absoluteDelta = 0.0
        var squaredDelta = 0.0
        for (i in 1 until values.size) {
            val delta = values[i] - values[i - 1]
            absoluteDelta += abs(delta)
            squaredDelta += delta * delta
        }
        return floatArrayOf(mean.toFloat(), sqrt(variance).toFloat(),
            (values.max() - values.min()).toFloat(),
            (absoluteDelta / (values.size - 1)).toFloat(),
            sqrt(squaredDelta / (values.size - 1)).toFloat(),
            (values.count { abs(it - 1.0) > 0.1 }.toDouble() / values.size).toFloat())
    }
}

/** Timestamped bounded buffer. No interpolation, zero filling, or last-value carry. */
class MotionWindow {
    private val capacity = 1920
    private val ids = LongArray(capacity) { Long.MIN_VALUE }
    private val sums = DoubleArray(capacity)
    private val counts = IntArray(capacity)
    private var lastTimestamp = Long.MIN_VALUE
    var rejectedSamples = 0L
        private set

    @Synchronized fun clear() {
        ids.fill(Long.MIN_VALUE)
        sums.fill(0.0)
        counts.fill(0)
        lastTimestamp = Long.MIN_VALUE
        rejectedSamples = 0
    }

    @Synchronized fun add(timestampNanos: Long, magnitudeG: Double): Boolean {
        if (timestampNanos < 0 || timestampNanos < lastTimestamp ||
            !magnitudeG.isFinite() || magnitudeG < 0) {
            rejectedSamples++
            return false
        }
        lastTimestamp = timestampNanos
        val bin = timestampNanos / MotionFeatures.BIN_NANOS
        val slot = (bin % capacity).toInt()
        if (ids[slot] != bin) {
            ids[slot] = bin
            sums[slot] = 0.0
            counts[slot] = 0
        }
        sums[slot] += magnitudeG
        counts[slot]++
        return true
    }

    @Synchronized fun isFresh(nowNanos: Long): Boolean =
        lastTimestamp != Long.MIN_VALUE && nowNanos - lastTimestamp in 0L..2_000_000_000L

    @Synchronized fun coverage(endExclusive: Long): Int =
        (endExclusive - MotionFeatures.WINDOW_BINS until endExclusive).count { bin ->
            bin >= 0 && ids[(bin % capacity).toInt()] == bin
        }

    @Synchronized fun sampleCount(endExclusive: Long): Int =
        (endExclusive - MotionFeatures.WINDOW_BINS until endExclusive).sumOf { bin ->
            if (bin >= 0 && ids[(bin % capacity).toInt()] == bin)
                counts[(bin % capacity).toInt()] else 0
        }

    @Synchronized fun complete(endExclusive: Long): DoubleArray? {
        if (coverage(endExclusive) != MotionFeatures.WINDOW_BINS) return null
        return DoubleArray(MotionFeatures.WINDOW_BINS) { i ->
            val slot = ((endExclusive - MotionFeatures.WINDOW_BINS + i) % capacity).toInt()
            sums[slot] / counts[slot]
        }
    }
}
