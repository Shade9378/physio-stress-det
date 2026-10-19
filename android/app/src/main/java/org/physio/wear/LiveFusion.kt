package org.physio.wear

import android.content.Context
import org.json.JSONObject

data class LivePrepared(val model: StressPredictor, val features: FloatArray,
    val metadata: JSONObject, val sensors: Set<String>, val quality: String,
    val ppgValue: Double?, val skinTemperature: Double?, val skinConductance: Double? = null)

/** Sensor loss, invalid measurements and sparse windows all make that input unavailable. */
class LiveFusion(private val context: Context, private val fallback: StressPredictor,
    private val fallbackMetadata: JSONObject) : java.io.Closeable {
    private val ppgWindow = LiveInputWindow(25)
    private val edaWindow = LiveInputWindow(1)
    private val temperatureWindow = LiveInputWindow(1)
    private val extractor = LiveFeatures(context)
    private val catalog: List<SensorModelSpec>
    private val sessions = mutableMapOf<String,SensorModel>()
    init {
        val root = JSONObject(context.assets.open("generated/live_models.json").bufferedReader().use { it.readText() })
        require(root.getInt("schema") == 1)
        val array = root.getJSONArray("models")
        catalog = (0 until array.length()).map { SensorModelSpec(array.getJSONObject(it)) }
    }
    fun add(sample: PhysiologicalSample) {
        when (sample.sensor) {
            "BVP" -> ppgWindow.add(sample.timestampNanos,sample.value,sample.valid)
            "EDA" -> edaWindow.add(sample.timestampNanos,sample.value,sample.valid && sample.value >= 0)
            "TEMP" -> temperatureWindow.add(sample.timestampNanos,sample.value,sample.valid)
        }
    }
    fun hasFresh(sensor: String, nowNanos: Long): Boolean {
        val latest = when (sensor) {
            "BVP" -> ppgWindow.latestTimestamp()
            "EDA" -> edaWindow.latestTimestamp()
            "TEMP" -> temperatureWindow.latestTimestamp()
            else -> return true
        } ?: return false
        return nowNanos-latest in 0L..5_000_000_000L
    }
    fun alignedEnd(accEndNanos: Long): Long {
        var end = accEndNanos
        ppgWindow.latestTimestamp()?.let { if (hasFresh("BVP",accEndNanos)) end = minOf(end,it+40_000_000L) }
        edaWindow.latestTimestamp()?.let { if (hasFresh("EDA",accEndNanos)) end = minOf(end,it+1_000_000_000L) }
        temperatureWindow.latestTimestamp()?.let { if (hasFresh("TEMP",accEndNanos)) end = minOf(end,it+1_000_000_000L) }
        return end
    }
    fun prepare(motionBins: DoubleArray, endNanos: Long): LivePrepared {
        val motion = MotionFeatures.extract(motionBins)
        val names = listOf("mean_g","std_g","range_g","mean_absolute_delta_g","rms_delta_g","fraction_dynamic")
        val values = names.zip(motion.map { it.toDouble() }).toMap().toMutableMap()
        val available = mutableSetOf("ACC")
        val reasons = mutableListOf<String>()
        val rawPpg = ppgWindow.complete(endNanos)
        val rawTemp = temperatureWindow.complete(endNanos)
        val rawEda = edaWindow.complete(endNanos)
        if (rawPpg != null) {
            runCatching { extractor.ppg(rawPpg) }.onSuccess {
                values.putAll(it); available.add("BVP")
            }.onFailure { reasons.add("PPG invalid") }
        } else reasons.add("PPG unavailable/incomplete")
        if (rawEda != null) {
            runCatching { extractor.eda(rawEda) }.onSuccess {
                values.putAll(it); available.add("EDA")
            }.onFailure { reasons.add("EDA invalid") }
        } else reasons.add("EDA unavailable/incomplete")
        if (rawTemp != null) {
            runCatching { extractor.temperature(rawTemp) }.onSuccess {
                values.putAll(it); available.add("TEMP")
            }.onFailure { reasons.add("Skin TEMP invalid") }
        } else reasons.add("Skin TEMP unavailable/incomplete")
        if (available == setOf("ACC")) return LivePrepared(fallback,motion,fallbackMetadata,available,
            reasons.joinToString(" · ") + " · ACC fallback",null,null)
        val spec = catalog.singleOrNull { it.sensors == available }
            ?: throw IllegalStateException("No validated live model for " + available.joinToString(" + "))
        val model = sessions.getOrPut(spec.case) { SensorModel(context,spec) }
        val features = spec.names.map { values.getValue(it).toFloat() }.toFloatArray()
        return LivePrepared(model,features,spec.metadata,available,
            ("Complete " + available.joinToString(" + ") + if (reasons.isEmpty()) "" else " · " + reasons.joinToString(" · ")),
            rawPpg?.lastOrNull(),rawTemp?.lastOrNull(),rawEda?.lastOrNull())
    }
    override fun close() { sessions.values.forEach { it.close() }; sessions.clear() }
}
