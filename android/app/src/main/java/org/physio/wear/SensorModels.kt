package org.physio.wear

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.nio.FloatBuffer
import java.security.MessageDigest
import org.json.JSONObject

/** Exact subset selection. No fabricated features or BPM-as-BVP substitution. */
object SensorPolicy {
    val all = setOf("ACC", "BVP", "EDA", "TEMP")
    fun select(available: Set<String>, contracts: Map<String, Set<String>>): String? {
        require(available.all { it in all })
        return contracts.entries.singleOrNull { it.value == available }?.key
    }
}

data class SensorModelSpec(val metadata: JSONObject) {
    val case: String = metadata.getString("case")
    val sensors: Set<String> = metadata.getJSONArray("sensors").let { a ->
        (0 until a.length()).map { a.getString(it) }.toSet()
    }
    val names: List<String> = metadata.getJSONArray("feature_names").let { a ->
        (0 until a.length()).map { a.getString(it) }
    }
    val id: String = metadata.getString("model_id")

    fun features(window: JSONObject): FloatArray {
        val values = window.getJSONObject("features")
        return names.map { values.getDouble(it).toFloat() }.toFloatArray().also {
            require(it.all { value -> value.isFinite() }) { "Missing/nonfinite sensor features" }
        }
    }
}

class SensorModelCatalog(context: Context) {
    val models: List<SensorModelSpec>
    init {
        val registry = JSONObject(context.assets.open("generated/sensor_models.json").bufferedReader().use { it.readText() })
        require(registry.getInt("schema") == 1 && registry.getInt("window_seconds") == 60)
        val array = registry.getJSONArray("models")
        models = (0 until array.length()).map { SensorModelSpec(array.getJSONObject(it)) }
        require(models.map { it.case }.distinct().size == models.size)
        require(models.all { it.sensors.isNotEmpty() && it.sensors.all { s -> s in SensorPolicy.all } &&
            it.names.size == it.metadata.getInt("input_count") && it.names.distinct().size == it.names.size })
    }
    fun select(available: Set<String>): SensorModelSpec? {
        val case = SensorPolicy.select(available, models.associate { it.case to it.sensors })
        return models.singleOrNull { it.case == case }
    }
}

/** ONNX Runtime 1.20.0; prepared feature order checked against the model contract. */
class SensorModel(context: Context, private val spec: SensorModelSpec) : StressPredictor {
    private val environment = OrtEnvironment.getEnvironment()
    private val session = OrtSession.SessionOptions().use { options ->
        options.setIntraOpNumThreads(1)
        options.setInterOpNumThreads(1)
        val file = spec.metadata.getString("file")
        require(file.matches(Regex("sensor_models/[a-z_]+\\.onnx")))
        val bytes = context.assets.open("generated/$file").use { it.readBytes() }
        val digest = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
        require(digest == spec.metadata.getString("model_sha256")) { "Model checksum mismatch" }
        environment.createSession(bytes, options).also { loaded ->
            require((loaded.inputInfo["features"]?.info as TensorInfo).shape.contentEquals(longArrayOf(-1, spec.names.size.toLong())))
            require((loaded.outputInfo["probability"]?.info as TensorInfo).shape.contentEquals(longArrayOf(-1, 1)))
        }
    }
    @Synchronized override fun predict(features: FloatArray): Float {
        require(features.size == spec.names.size && features.all { it.isFinite() })
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(features), longArrayOf(1, features.size.toLong())).use { input ->
            session.run(mapOf("features" to input)).use { output ->
                val probability = (output[0] as OnnxTensor).floatBuffer.get(0)
                require(probability.isFinite() && probability in -1e-6f..1.000001f) {
                    "Invalid probability for ${spec.case}: $probability"
                }
                // Mobile sigmoid kernels can overshoot endpoints by a few float32 ULPs.
                return probability.coerceIn(0f, 1f)
            }
        }
    }
    @Synchronized override fun close() { session.close() }
}
