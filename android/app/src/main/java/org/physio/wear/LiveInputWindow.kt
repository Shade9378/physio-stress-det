package org.physio.wear

import java.util.ArrayDeque

data class PhysiologicalSample(val sensor: String, val timestampNanos: Long, val value: Double, val valid: Boolean)

/** No extrapolation or long-gap filling. Linear alignment only between adjacent samples <=2 periods apart. */
class LiveInputWindow(private val hz: Int) {
    private data class Sample(val time: Long, val value: Double)
    private val values = ArrayDeque<Sample>()
    private val period = 1_000_000_000L/hz
    init { require(hz in listOf(1,25)) }
    @Synchronized fun latestTimestamp(): Long? = values.peekLast()?.time
    @Synchronized fun clear() { values.clear() }
    @Synchronized fun add(time: Long, value: Double, valid: Boolean): Boolean {
        if (!valid || !value.isFinite()) { clear(); return false }
        if (time < 0 || (values.peekLast()?.time ?: -1) >= time) return false
        values.add(Sample(time,value))
        while (values.size > hz*65 || values.first.time < time-65_000_000_000L) values.removeFirst()
        return true
    }
    @Synchronized fun complete(endNanos: Long): DoubleArray? {
        val samples = values.toList()
        if (samples.isEmpty() || endNanos-samples.last().time > 2*period) return null
        val start = endNanos-60_000_000_000L
        if (start < 0 || samples.first().time > start) return null
        var index = 0
        val result = DoubleArray(hz*60)
        for (i in result.indices) {
            val target = start+i*period
            while (index+1 < samples.size && samples[index+1].time <= target) index++
            val a = samples[index]
            if (a.time == target) result[i] = a.value else {
                if (index+1 >= samples.size) return null
                val b = samples[index+1]
                if (b.time-a.time > 2*period || target !in a.time..b.time) return null
                result[i] = a.value+(b.value-a.value)*(target-a.time).toDouble()/(b.time-a.time)
            }
        }
        return result
    }
}

interface VendorSensorSource : java.io.Closeable { fun start() }

/** Identifies SDK milliseconds only when a documented Android clock can be matched. */
class VendorClock {
    private var wallOffset: Long? = null
    fun convert(timestampMillis: Long, wallMillis: Long, elapsedNanos: Long): Long? {
        if (timestampMillis < 0) return null
        val elapsedMillis = elapsedNanos/1_000_000
        val asElapsed = elapsedMillis-timestampMillis in -1000L..65_000L
        val asWall = wallMillis-timestampMillis in -1000L..65_000L
        if (asElapsed == asWall) return null
        if (asElapsed) return timestampMillis*1_000_000
        val offset = wallMillis-elapsedMillis
        if (wallOffset != null && kotlin.math.abs(wallOffset!!-offset) > 500) return null
        wallOffset = offset
        return (timestampMillis-offset)*1_000_000
    }
}
