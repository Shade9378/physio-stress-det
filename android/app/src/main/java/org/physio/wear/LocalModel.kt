package org.physio.wear

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.security.MessageDigest
import org.json.JSONObject
import ai.onnxruntime.TensorInfo

class LocalModel(context: Context) : StressPredictor {
    private val environment = OrtEnvironment.getEnvironment()
    private val session = OrtSession.SessionOptions().use { options ->
        options.setIntraOpNumThreads(1)
        options.setInterOpNumThreads(1)
        val metadata = JSONObject(context.assets.open("generated/model.json").bufferedReader().use { it.readText() })
        require(metadata.getInt("schema") == 2 && metadata.getInt("bin_hz") == 16 && metadata.getInt("window_seconds") == 60)
        require(metadata.getString("model_family") == "mlp" && metadata.getString("model_id").isNotBlank())
        val names = listOf("mean_g", "std_g", "range_g", "mean_absolute_delta_g", "rms_delta_g", "fraction_dynamic")
        val expected = metadata.getJSONArray("feature_names")
        require((0 until expected.length()).map { expected.getString(it) } == names)
        val bytes = context.assets.open("generated/stress.onnx").use { it.readBytes() }
        val digest = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
        require(digest == metadata.getString("model_sha256")) { "Model checksum mismatch" }
        environment.createSession(bytes, options).also { loaded ->
            require((loaded.inputInfo["features"]?.info as TensorInfo).shape.contentEquals(longArrayOf(-1, 6)))
            require((loaded.outputInfo["probability"]?.info as TensorInfo).shape.contentEquals(longArrayOf(-1, 1)))
        }
    }

    @Synchronized override fun predict(features: FloatArray): Float {
        require(features.size == 6 && features.all { it.isFinite() })
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(features), longArrayOf(1, 6)).use { input ->
            session.run(mapOf("features" to input)).use { output ->
                val probability = (output[0] as OnnxTensor).floatBuffer.get(0)
                require(probability.isFinite() && probability in 0f..1f)
                return probability
            }
        }
    }

    @Synchronized override fun close() { session.close() }
}
