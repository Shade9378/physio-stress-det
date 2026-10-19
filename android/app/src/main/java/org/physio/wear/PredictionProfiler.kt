package org.physio.wear

import java.util.concurrent.atomic.AtomicLong

interface StressPredictor : java.io.Closeable { fun predict(features: FloatArray): Float }

data class PredictionTiming(val id: Long, val timestamps: List<Long>) {
    val stageMilliseconds: List<Double> get() = timestamps.zipWithNext().map { (a,b) -> (b-a)/1e6 }
    val totalMilliseconds: Double get() = (timestamps.last()-timestamps.first())/1e6
}

/** Optional software trace. No sensor data or background recording in normal app operation. */
class PredictionProfiler(val enabled: Boolean = false) {
    private val identifiers = AtomicLong()
    private val submitted = linkedMapOf<Long, List<Long>>()
    private val registered = mutableSetOf<Long>()
    private val committed = linkedMapOf<Long, PredictionTiming>()

    @Synchronized fun publish(timestamps: List<Long>): Long {
        if (!enabled) return 0
        require(timestamps.size == 6 && timestamps.zipWithNext().all { (a,b) -> b >= a })
        val id = identifiers.incrementAndGet()
        submitted[id] = timestamps.toList()
        return id
    }

    @Synchronized fun registerDraw(id: Long): Boolean =
        enabled && id in submitted && id !in committed && registered.add(id)

    @Synchronized fun commitFrame(id: Long, committedAt: Long) {
        val values = submitted[id] ?: return
        if (id in committed) return
        require(committedAt >= values.last())
        committed[id] = PredictionTiming(id, values + committedAt)
    }

    @Synchronized fun submittedCount(): Int = submitted.size
    @Synchronized fun snapshot(): List<PredictionTiming> = committed.values.toList()
    @Synchronized fun pendingIds(): List<Long> = submitted.keys.filter { it !in committed }
}
