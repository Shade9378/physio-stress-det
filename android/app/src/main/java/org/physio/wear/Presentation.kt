package org.physio.wear

import org.json.JSONObject
import java.util.Locale

fun sensorLabel(sensor: String): String = when (sensor) {
    "ACC" -> "Movement"
    "BVP" -> "Pulse"
    "EDA" -> "Skin response"
    "TEMP" -> "Skin temperature"
    else -> sensor
}

fun homeStatus(state: MonitorState): String {
    if (state.status.startsWith("Cannot run:")) {
        return when {
            state.status.contains("Accelerometer", ignoreCase = true) -> "Motion sensor unavailable"
            listOf("model", "onnx", "generated/", "checksum").any {
                state.status.contains(it, ignoreCase = true)
            } -> "Model unavailable. Please reinstall the app."
            state.status.contains("Replay data not installed") -> "Replay unavailable. Choose Watch in Settings."
            else -> "Unable to start. Please try again."
        }
    }
    if (!state.running) {
        return when {
            state.status == "Replay finished" -> "Session complete"
            state.probability != null -> "Monitoring stopped"
            else -> "Tap Start to begin"
        }
    }
    if (state.probability != null) return "Monitoring"
    if (listOf("stale", "unreliable", "delayed", "invalid", "no recent").any {
            state.quality.contains(it, ignoreCase = true)
        }) return "Waiting for clearer readings"
    val seconds = if (state.mode == "Replay" && state.coverage == 0)
        state.elapsedSeconds.toInt().coerceIn(0, 60) else (state.coverage / 16).coerceIn(0, 60)
    return if (seconds < 60) "Collecting readings · $seconds/60 s" else "Waiting for a reading"
}

/** Only fields supplied by the currently selected model; no fixed architecture or study text. */
fun modelInformation(metadata: JSONObject): List<Pair<String, String>> = buildList {
    val identifier = metadata.optString("model_id")
    if (identifier.isNotBlank()) add("Model ID" to identifier.replace("\u2192", "->"))
    val sensors = metadata.optJSONArray("sensors")
    if (sensors != null && sensors.length() > 0) {
        add("Sensors" to (0 until sensors.length()).joinToString(", ") {
            sensorLabel(sensors.getString(it))
        })
    }
    metadata.optJSONArray("feature_names")?.let { add("Input features" to it.length().toString()) }
    metadata.optJSONArray("evaluation")?.optJSONObject(0)?.let { evaluation ->
        val mean = evaluation.optDouble("macro_f1_mean")
        val deviation = evaluation.optDouble("macro_f1_std")
        if (mean.isFinite()) {
            add("Validation F1" to if (deviation.isFinite())
                String.format(Locale.US, "%.3f +/- %.3f", mean, deviation)
            else String.format(Locale.US, "%.3f", mean))
        }
    }
}


/** Display measured values; derived replay pulse rate is never used as a raw BVP substitute. */
fun supportingReadings(state: MonitorState): List<Pair<String, String>> {
    fun value(number: Double?, format: String): String =
        number?.takeIf { it.isFinite() }?.let { String.format(Locale.US, format, it) }
            ?: "Unavailable"
    val pulse = state.heartRate?.takeIf { it.isFinite() && it > 0 } ?: state.pulseRateBpm
    return listOf(
        "Pulse rate" to value(pulse, "%.0f bpm"),
        "Skin temperature" to value(state.temperature, "%.1f °C"),
        "Movement" to value(state.magnitudeG, "%.2f g"),
        "Skin response" to value(state.eda, "%.2f µS")
    )
}
