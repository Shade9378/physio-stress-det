package org.physio.wear

import android.content.Context
import android.os.SystemClock
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.*
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicLong

data class MonitorState(
    val running: Boolean = false, val mode: String = "Replay", val status: String = "Ready",
    val probability: Float? = null, val magnitudeG: Double? = null, val heartRate: Double? = null,
    val heartStatus: String = "Not requested", val quality: String = "Waiting for samples",
    val coverage: Int = 0, val latencyMs: Double? = null, val throughputHz: Double = 0.0,
    val elapsedSeconds: Double = 0.0, val sensorAvailable: Boolean = false,
    val history: List<Estimate> = emptyList(), val sessionCount: Int = 0, val sessionId: Long = 0,
    val predictionTraceId: Long = 0, val inputs: String = "ACC + BVP + EDA + TEMP",
    val sensorStatus: String = "Replay: all four WESAD sensors",
    val bvp: Double? = null, val eda: Double? = null, val temperature: Double? = null,
    val pulseRateBpm: Double? = null
)

private data class AccSample(val timestamp: Long, val magnitude: Double, val reliable: Boolean, val receivedNanos: Long)

/** Inference/acquisition/storage off the main thread. Never synthesizes missing input. */
class MonitoringEngine(private val context: Context,
    val profiler: PredictionProfiler = PredictionProfiler(),
    private val predictorFactory: (() -> StressPredictor)? = null,
    private val historyDatabase: String = "stress.db",
    private val benchmarkModelId: String? = null
) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val history = HistoryStore(context, historyDatabase)
    private val mutable = MutableStateFlow(MonitorState())
    val state: StateFlow<MonitorState> = mutable.asStateFlow()
    private var job: Job? = null
    private var sensors: LiveSensors? = null
    private val generation = AtomicLong()
    private val accMetadata: JSONObject = runCatching {
        JSONObject(context.assets.open("generated/model.json").bufferedReader().use { it.readText() })
    }.getOrDefault(JSONObject())

    val replayAvailable: Boolean get() = "sensor_replay.json" in context.assets.list("generated").orEmpty()

    private val catalog by lazy { SensorModelCatalog(context) }
    @Volatile private var selectedMetadata: JSONObject? = null
    val metadata: JSONObject get() = selectedMetadata ?: runCatching {
        catalog.select(SensorPolicy.all)?.metadata
    }.getOrNull() ?: accMetadata

    val modelId: String get() = benchmarkModelId ?: metadata.optString("model_id", "unknown")

    init { scope.launch { mutable.update { it.copy(history = history.recent()) } } }

    fun start(mode: String, intervalSeconds: Int, requestedHz: Int, replaySpeed: Int,
        replaySensors: Set<String> = SensorPolicy.all) {
        stop()
        val token = generation.incrementAndGet()
        mutable.update { MonitorState(running = true, mode = mode, status = "Loading local model",
            history = it.history, sessionId = System.currentTimeMillis()) }
        job = scope.launch {
            try {
                if (mode == "Replay" && predictorFactory == null)
                    replayPrepared(intervalSeconds, replaySpeed, token, replaySensors)
                else {
                    selectedMetadata = accMetadata
                    (predictorFactory?.invoke() ?: LocalModel(context)).use { model ->
                        if (mode == "Replay") replay(model, intervalSeconds, replaySpeed, token)
                        else live(model, intervalSeconds, requestedHz, token)
                    }
                }
            } catch (cancelled: CancellationException) { throw cancelled }
            catch (error: Exception) {
                if (generation.get() == token) mutable.update { it.copy(running = false,
                    probability = null, status = "Cannot run: " + error.message) }
            } finally {
                if (generation.get() == token) {
                    sensors?.close()
                    sensors = null
                    mutable.update { it.copy(running = false) }
                }
            }
        }
    }

    /** Prepared WESAD windows, not raw sensor streaming or native full-sensor preprocessing. */
    private suspend fun replayPrepared(interval: Int, speed: Int, token: Long, enabled: Set<String>) {
        require(interval in listOf(5, 10) && speed > 0)
        require(replayAvailable) { "Replay data not installed. Use Live mode or export local WESAD assets." }
        val replay = JSONObject(context.assets.open("generated/sensor_replay.json").bufferedReader().use { it.readText() })
        val segments = replay.getJSONArray("segments")
        var current: SensorModelSpec? = null
        var model: SensorModel? = null
        try {
            for (segmentIndex in 0 until segments.length()) {
                val segment = segments.getJSONObject(segmentIndex)
                val windows = segment.getJSONArray("windows")
                if (generation.get() != token) return
                mutable.update { it.copy(probability = null, coverage = 0,
                    quality = "Prepared 60 s windows · 10 s source stride",
                    status = "Replayed WESAD / " + if (segment.getInt("target") == 0) "baseline" else "stress",
                    heartRate = null, heartStatus = "BPM not supplied; raw BVP is separate") }
                val started = SystemClock.elapsedRealtimeNanos()
                for (index in 0 until windows.length()) {
                    currentCoroutineContext().ensureActive()
                    // Source windows are 10 s apart. A 5 s setting cannot create extra labeled windows.
                    val sourceEnd = 60 + index * 10
                    val deadline = started + sourceEnd * 1_000_000_000L / speed
                    while (SystemClock.elapsedRealtimeNanos() < deadline) {
                        delay(minOf(100L, maxOf(1L, (deadline - SystemClock.elapsedRealtimeNanos()) / 1_000_000)))
                        if (generation.get() != token) return
                        mutable.update { it.copy(elapsedSeconds = minOf(sourceEnd.toDouble(),
                            (SystemClock.elapsedRealtimeNanos() - started) / 1e9 * speed)) }
                    }
                    val window = windows.getJSONObject(index)
                    val available = window.getJSONArray("available_sensors").let { a ->
                        (0 until a.length()).map { a.getString(it) }.toSet()
                    }.intersect(enabled)
                    val spec = catalog.select(available)
                    if (spec == null) {
                        model?.close(); model = null; current = null
                        mutable.update { it.copy(probability = null, inputs = available.joinToString(" + "),
                            quality = "No validated model for available inputs · classification disabled") }
                        continue
                    }
                    if (current?.case != spec.case) {
                        model?.close()
                        model = SensorModel(context, spec)
                        current = spec
                        selectedMetadata = spec.metadata
                    }
                    val values = window.getJSONObject("features")
                    mutable.update { it.copy(inputs = spec.sensors.joinToString(" + "), coverage = 960,
                        sensorStatus = "Prepared replay: " + spec.sensors.joinToString(" + "),
                        magnitudeG = if ("ACC" in available) values.getDouble("mean_g") else null,
                        bvp = if ("BVP" in available) values.getDouble("bvp_0_mean") else null,
                        pulseRateBpm = if ("BVP" in available)
                            values.optDouble("bvp_heart_rate").takeIf { it.isFinite() && it > 0 } else null,
                        eda = if ("EDA" in available) values.getDouble("eda_0_mean") else null,
                        temperature = if ("TEMP" in available) values.getDouble("temp_0_mean") else null,
                        quality = "Complete prepared window · local inference", elapsedSeconds = sourceEnd.toDouble()) }
                    val received = SystemClock.elapsedRealtimeNanos()
                    val features = spec.features(window)
                    estimatePrepared(model!!, features, "Replay", window.getInt("target"), token, received, received)
                }
            }
            if (generation.get() == token) mutable.update { it.copy(status = "Replay finished") }
        } finally { model?.close() }
    }

    private suspend fun replay(model: StressPredictor, interval: Int, speed: Int, token: Long) {
        val replay = JSONObject(context.assets.open("generated/replay.json").bufferedReader().use { it.readText() })
        val segments = replay.getJSONArray("segments")
        for (segmentIndex in 0 until segments.length()) {
            val segment = segments.getJSONObject(segmentIndex)
            val values = segment.getJSONArray("magnitude_g")
            val raw = segment.optJSONArray("acc_raw")
            val rate = if (raw == null) 16 else 32
            val samples = raw?.length() ?: values.length()
            val target = segment.getInt("target")
            val buffer = MotionWindow()
            if (generation.get() != token) return
            mutable.update { it.copy(status = "Replayed S2 / " + if (target == 0) "baseline" else "stress",
                coverage = 0, probability = null, quality = "New segment · collecting 60 s",
                heartRate = null, heartStatus = "Not supplied by replay") }
            val started = SystemClock.elapsedRealtimeNanos()
            for (i in 0 until samples) {
                currentCoroutineContext().ensureActive()
                val deadline = started + ((i + 1) * (1_000_000_000L / rate) / speed)
                val waitMillis = (deadline - SystemClock.elapsedRealtimeNanos()) / 1_000_000
                if (waitMillis > 0) delay(waitMillis)
                val received = SystemClock.elapsedRealtimeNanos()
                val magnitude = if (raw == null) values.getDouble(i) else {
                    val axes = raw.getJSONArray(i)
                    val x = axes.getDouble(0) / 64.0
                    val y = axes.getDouble(1) / 64.0
                    val z = axes.getDouble(2) / 64.0
                    kotlin.math.sqrt(x*x + y*y + z*z)
                }
                buffer.add(i * (1_000_000_000L / rate), magnitude)
                if (raw != null && (i + 1) % 2 != 0) continue
                val end = if (raw == null) i + 1L else (i + 1L) / 2
                if (end % 4L == 0L && generation.get() == token) mutable.update { it.copy(
                    magnitudeG = magnitude, coverage = buffer.coverage(end),
                    elapsedSeconds = end / 16.0, throughputHz = 16.0,
                    quality = if (end >= 960) "Complete replay input" else "Collecting 60 s") }
                if (end >= 960 && (end - 960) % (interval * 16) == 0L)
                    estimate(model, buffer.complete(end)!!, "Replay", target, token, received)
            }
        }
        if (generation.get() == token) mutable.update { it.copy(status = "Replay finished") }
    }

    private suspend fun live(model: StressPredictor, interval: Int, requestedHz: Int, token: Long) {
        val queue = Channel<AccSample>(Channel.BUFFERED)
        val dropped = AtomicLong()
        val lastHeartTimestamp = AtomicLong()
        val buffer = MotionWindow()
        val fusion = LiveFusion(context, model, accMetadata)
        val acquisition = LiveSensors(context, { timestamp, value, reliable ->
            if (!queue.trySend(AccSample(timestamp, value, reliable, SystemClock.elapsedRealtimeNanos())).isSuccess) dropped.incrementAndGet()
        }, { bpm, status ->
            lastHeartTimestamp.set(if (bpm == null) 0 else SystemClock.elapsedRealtimeNanos())
            if (generation.get() == token) mutable.update { it.copy(heartRate = bpm, heartStatus = status) }
        }, { sample ->
            fusion.add(sample)
            if (!sample.valid && generation.get() == token) mutable.update { it.copy(probability = null,
                bvp = if (sample.sensor == "BVP") null else it.bvp,
                eda = if (sample.sensor == "EDA") null else it.eda,
                temperature = if (sample.sensor == "TEMP") null else it.temperature,
                quality = sample.sensor + " invalid · awaiting subset inference") }
        }, { status ->
            if (generation.get() == token) mutable.update { it.copy(sensorStatus = status) }
        })
        sensors = acquisition
        if (!acquisition.start(requestedHz)) {
            acquisition.close()
            throw IllegalStateException("Accelerometer unavailable; classification disabled")
        }
        mutable.update { it.copy(sensorAvailable = true, inputs = "ACC", bvp = null, eda = null, temperature = null,
            status = if (android.os.Build.FINGERPRINT.contains("sdk") || android.os.Build.FINGERPRINT.contains("emu")) "Emulator sensors / no real physiology" else "Live / experimental transfer") }
        val watchdog = scope.launch {
            while (isActive && generation.get() == token) {
                delay(1000)
                val now = SystemClock.elapsedRealtimeNanos()
                val staleAcceleration = !buffer.isFresh(now)
                val lastHeart = lastHeartTimestamp.get()
                val staleHeart = lastHeart > 0 && now - lastHeart > 15_000_000_000L
                val stalePhysiology = ("BVP" in mutable.value.inputs && !fusion.hasFresh("BVP",now)) ||
                    ("TEMP" in mutable.value.inputs && !fusion.hasFresh("TEMP",now)) ||
                    ("EDA" in mutable.value.inputs && !fusion.hasFresh("EDA",now))
                if (generation.get() == token) mutable.update { it.copy(
                    probability = if (staleAcceleration || stalePhysiology) null else it.probability,
                    coverage = if (staleAcceleration) 0 else it.coverage,
                    bvp = if (!fusion.hasFresh("BVP",now)) null else it.bvp,
                    eda = if (!fusion.hasFresh("EDA",now)) null else it.eda,
                    temperature = if (!fusion.hasFresh("TEMP",now)) null else it.temperature,
                    quality = if (staleAcceleration) "No recent acceleration · estimates disabled" else if (stalePhysiology) "Physiology stale · awaiting subset inference" else it.quality,
                    heartRate = if (staleHeart) null else it.heartRate,
                    heartStatus = if (staleHeart) "No recent BPM / optional" else it.heartStatus) }
            }
        }
        var firstTimestamp = -1L
        var lastInferenceEnd = -1L
        var lastUiBin = -1L
        try {
            for (sample in queue) {
                if (generation.get() != token) return
                if (firstTimestamp < 0) firstTimestamp = sample.timestamp
                if (!sample.reliable) {
                    buffer.clear()
                    mutable.update { it.copy(probability = null, coverage = 0, quality = "Unreliable sensor · estimates disabled") }
                    continue
                }
                buffer.add(sample.timestamp, sample.magnitude)
                if (!buffer.isFresh(SystemClock.elapsedRealtimeNanos())) {
                    mutable.update { it.copy(probability = null, coverage = 0,
                        quality = "Delayed acceleration · estimates disabled") }
                    continue
                }
                val end = sample.timestamp / MotionFeatures.BIN_NANOS
                val count = buffer.coverage(end)
                val hz = buffer.sampleCount(end) / 60.0
                if (end - lastUiBin >= 4) {
                    lastUiBin = end
                    val lastHeart = lastHeartTimestamp.get()
                    val staleHeart = lastHeart > 0 && SystemClock.elapsedRealtimeNanos() - lastHeart > 15_000_000_000L
                    mutable.update { it.copy(magnitudeG = sample.magnitude, coverage = count,
                        heartRate = if (staleHeart) null else it.heartRate,
                        heartStatus = if (staleHeart) "No recent BPM / optional" else it.heartStatus,
                        throughputHz = hz, elapsedSeconds = (sample.timestamp - firstTimestamp) / 1e9,
                        quality = when {
                            count < 960 -> "Incomplete window · classification disabled"
                            hz < 24 -> "Sampling too sparse · classification disabled"
                            else -> "Complete input · dropped queue samples: " + dropped.get()
                        }, probability = if (count < 960 || hz < 24) null else it.probability) }
                }
                if (count == 960 && hz >= 24 &&
                    (lastInferenceEnd < 0 || end - lastInferenceEnd >= interval * 16)) {
                    val alignedEnd = fusion.alignedEnd(end * MotionFeatures.BIN_NANOS) / MotionFeatures.BIN_NANOS
                    if (lastInferenceEnd >= 0 && alignedEnd-lastInferenceEnd < interval*16) continue
                    val values = buffer.complete(alignedEnd) ?: continue
                    val ready = SystemClock.elapsedRealtimeNanos()
                    val prepared = fusion.prepare(values, alignedEnd * MotionFeatures.BIN_NANOS)
                    selectedMetadata = prepared.metadata
                    mutable.update { it.copy(inputs = prepared.sensors.joinToString(" + "),
                        bvp = prepared.ppgValue, temperature = prepared.skinTemperature, eda = prepared.skinConductance, quality = prepared.quality) }
                    estimatePrepared(prepared.model, prepared.features, "Live", null, token, sample.receivedNanos, ready)
                    lastInferenceEnd = alignedEnd
                }
            }
        } finally { watchdog.cancel(); acquisition.close(); fusion.close(); queue.close() }
    }

    private suspend fun estimate(model: StressPredictor, bins: DoubleArray, mode: String,
        target: Int?, token: Long, receivedNanos: Long) {
        val ready = SystemClock.elapsedRealtimeNanos()
        val features = MotionFeatures.extract(bins)
        estimatePrepared(model, features, mode, target, token, receivedNanos, ready)
    }

    private suspend fun estimatePrepared(model: StressPredictor, features: FloatArray, mode: String,
        target: Int?, token: Long, receivedNanos: Long, ready: Long) {
        val featuresReady = SystemClock.elapsedRealtimeNanos()
        val probability = model.predict(features)
        val inferenceDone = SystemClock.elapsedRealtimeNanos()
        val elapsed = (inferenceDone - ready) / 1e6
        currentCoroutineContext().ensureActive()
        if (generation.get() != token) return
        val record = Estimate(System.currentTimeMillis(), probability, mode, elapsed, target, mutable.value.sessionId, modelId)
        history.insert(record)
        val recent = history.recent()
        val storageDone = SystemClock.elapsedRealtimeNanos()
        if (generation.get() != token) return
        val published = SystemClock.elapsedRealtimeNanos()
        val traceId = profiler.publish(listOf(receivedNanos, ready, featuresReady, inferenceDone, storageDone, published))
        mutable.update { it.copy(probability = probability, latencyMs = elapsed,
            history = recent, sessionCount = it.sessionCount + 1, predictionTraceId = traceId) }
    }

    fun stop() {
        generation.incrementAndGet()
        job?.cancel()
        job = null
        sensors?.close()
        sensors = null
        mutable.update { it.copy(running = false, status = "Stopped", heartRate = null) }
    }
    fun clearHistory() { scope.launch { history.clear(); mutable.update { it.copy(history = emptyList()) } } }
    fun close() { stop(); scope.cancel() }
}
