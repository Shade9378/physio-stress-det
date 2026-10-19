package org.physio.wear

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import androidx.health.services.client.HealthServices
import androidx.health.services.client.MeasureCallback
import androidx.health.services.client.data.Availability
import androidx.health.services.client.data.DataPointContainer
import androidx.health.services.client.data.DataType
import androidx.health.services.client.data.DeltaDataType
import java.util.concurrent.Executors
import kotlin.math.sqrt

/** Standard Wear OS plus optional, explicitly version-pinned vendor providers. */
class LiveSensors(private val context: Context,
    private val onAcceleration: (Long, Double, Boolean) -> Unit,
    private val onHeartRate: (Double?, String) -> Unit,
    private val onPhysiology: (PhysiologicalSample) -> Unit = {},
    private val onSensorStatus: (String) -> Unit = {}) : SensorEventListener {
    private val manager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val sensor = manager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val thread = HandlerThread("physio-sensors").apply { start() }
    private val executor = Executors.newSingleThreadExecutor()
    private val measure = HealthServices.getClient(context).measureClient
    @Volatile private var active = false
    @Volatile private var reliable = true
    private var heartRegistered = false
    private var vendors = emptyList<VendorSensorSource>()
    val available: Boolean get() = sensor != null
    private val heartCallback = object : MeasureCallback {
        override fun onRegistrationFailed(throwable: Throwable) {
            if (active) onHeartRate(null, "Registration failed / optional")
        }
        override fun onAvailabilityChanged(dataType: DeltaDataType<*, *>, availability: Availability) {
            if (active) onHeartRate(null, availability.toString())
        }
        override fun onDataReceived(data: DataPointContainer) {
            if (active) data.getData(DataType.HEART_RATE_BPM).lastOrNull()?.let {
                onHeartRate(it.value, "Available · display only")
            }
        }
    }

    fun start(requestedHz: Int): Boolean {
        active = true
        vendors = VendorProviders.create(context,onPhysiology,onSensorStatus)
        vendors.forEach { it.start() }
        val registered = sensor?.let {
            manager.registerListener(this, it, 1_000_000 / requestedHz, Handler(thread.looper))
        } ?: false
        if (context.checkSelfPermission(Manifest.permission.BODY_SENSORS) != PackageManager.PERMISSION_GRANTED) {
            onHeartRate(null, "Permission denied / optional")
        } else {
            try {
                val future = measure.getCapabilitiesAsync()
                future.addListener({
                    try {
                        val capabilities = future.get()
                        synchronized(this@LiveSensors) {
                            if (active && DataType.HEART_RATE_BPM in capabilities.supportedDataTypesMeasure) {
                                measure.registerMeasureCallback(DataType.HEART_RATE_BPM, executor, heartCallback)
                                heartRegistered = true
                            } else if (active) onHeartRate(null, "Unavailable on this device")
                        }
                    } catch (error: Exception) { if (active) onHeartRate(null, "Unavailable: " + error.javaClass.simpleName) }
                }, executor)
            } catch (error: Exception) { onHeartRate(null, "Health Services unavailable") }
        }
        return registered
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (!active || event.sensor.type != Sensor.TYPE_ACCELEROMETER) return
        val magnitude = sqrt(event.values.take(3).sumOf { it.toDouble() * it.toDouble() }) / MotionFeatures.GRAVITY
        onAcceleration(event.timestamp, magnitude, reliable)
    }
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        reliable = accuracy != SensorManager.SENSOR_STATUS_UNRELIABLE
    }
    @Synchronized fun close() {
        if (!active && executor.isShutdown) return
        active = false
        vendors.forEach { runCatching { it.close() } }
        vendors = emptyList()
        manager.unregisterListener(this)
        if (heartRegistered) runCatching {
            measure.unregisterMeasureCallbackAsync(DataType.HEART_RATE_BPM, heartCallback)
        }
        heartRegistered = false
        thread.quitSafely()
        executor.shutdown()
    }
}
