package org.physio.wear

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.SystemClock
import com.samsung.android.service.health.tracking.ConnectionListener
import com.samsung.android.service.health.tracking.HealthTracker
import com.samsung.android.service.health.tracking.HealthTrackerException
import com.samsung.android.service.health.tracking.HealthTrackingService
import com.samsung.android.service.health.tracking.data.DataPoint
import com.samsung.android.service.health.tracking.data.HealthTrackerType
import com.samsung.android.service.health.tracking.data.PpgType
import com.samsung.android.service.health.tracking.data.ValueKey

/** Optional Samsung Health Sensor SDK 1.4.1 bindings; all trackers require capability checks. */
class SamsungProvider : VendorProvider {
    override fun create(context: Context, onSample: (PhysiologicalSample) -> Unit,
        onStatus: (String) -> Unit): VendorSensorSource = SamsungSensors(context,onSample,onStatus)
}

private class SamsungSensors(private val context: Context,
    private val onSample: (PhysiologicalSample) -> Unit,
    private val onStatus: (String) -> Unit) : VendorSensorSource {
    private val trackers = mutableMapOf<String,HealthTracker>()
    private val clock = VendorClock()
    @Volatile private var active = false
    private val connection: ConnectionListener = object : ConnectionListener {
        override fun onConnectionSuccess() {
            synchronized(this@SamsungSensors) {
                if (!active) return
                val supported = service.trackingCapability.supportHealthTrackerTypes
                if (HealthTrackerType.PPG_CONTINUOUS in supported)
                    register("BVP") { service.getHealthTracker(HealthTrackerType.PPG_CONTINUOUS,setOf(PpgType.GREEN)) }
                if (HealthTrackerType.SKIN_TEMPERATURE_CONTINUOUS in supported)
                    register("TEMP") { service.getHealthTracker(HealthTrackerType.SKIN_TEMPERATURE_CONTINUOUS) }
                if (HealthTrackerType.EDA_CONTINUOUS in supported)
                    register("EDA") { service.getHealthTracker(HealthTrackerType.EDA_CONTINUOUS) }
                onStatus("Samsung 1.4.1 connected: " + trackers.keys.joinToString(" + ") +
                    " · needs complete, valid sensor windows")
            }
        }
        override fun onConnectionEnded() {
            invalidate("Samsung service disconnected")
        }
        override fun onConnectionFailed(error: HealthTrackerException) {
            invalidate("Samsung connection failed: " + error.errorCode)
        }
    }
    private val service: HealthTrackingService = HealthTrackingService(connection,context)
    private fun register(sensor: String, create: () -> HealthTracker) {
        try {
            val tracker = create()
            trackers[sensor] = tracker
            tracker.setEventListener(object : HealthTracker.TrackerEventListener {
                override fun onDataReceived(points: List<DataPoint>) {
                    synchronized(this@SamsungSensors) {
                        if (!active) return
                        for (point in points) {
                            try {
                                val value = when (sensor) {
                                    "BVP" -> point.getValue(ValueKey.PpgSet.PPG_GREEN).toDouble()
                                    "EDA" -> point.getValue(ValueKey.EdaSet.SKIN_CONDUCTANCE).toDouble()
                                    else -> point.getValue(ValueKey.SkinTemperatureSet.OBJECT_TEMPERATURE).toDouble()
                                }
                                val valid = when (sensor) {
                                    "BVP" -> point.getValue(ValueKey.PpgSet.GREEN_STATUS) == 0
                                    "EDA" -> point.getValue(ValueKey.EdaSet.STATUS) == 0 && value >= 0
                                    else -> point.getValue(ValueKey.SkinTemperatureSet.STATUS) == 0
                                }
                                val time = clock.convert(point.timestamp,System.currentTimeMillis(),SystemClock.elapsedRealtimeNanos())
                                onSample(PhysiologicalSample(sensor,time ?: 0,value,valid && time != null))
                            } catch (error: Exception) {
                                onSample(PhysiologicalSample(sensor,0,0.0,false))
                                onStatus("$sensor sample rejected: " + error.javaClass.simpleName)
                            }
                        }
                    }
                }
                override fun onError(error: HealthTracker.TrackerError) {
                    onSample(PhysiologicalSample(sensor,0,0.0,false))
                    onStatus("$sensor unavailable: $error")
                }
                override fun onFlushCompleted() {}
            })
        } catch (error: Exception) { onStatus("$sensor registration unavailable: " + error.javaClass.simpleName) }
    }
    @Synchronized private fun invalidate(reason: String) {
        for (sensor in listOf("BVP","EDA","TEMP")) onSample(PhysiologicalSample(sensor,0,0.0,false))
        onStatus(reason)
    }
    @Synchronized override fun start() {
        if (context.checkSelfPermission(Manifest.permission.BODY_SENSORS) != PackageManager.PERMISSION_GRANTED) {
            onStatus("Samsung BODY_SENSORS permission denied"); return
        }
        active = true
        try { service.connectService() }
        catch (error: Exception) { invalidate("Samsung service unavailable: " + error.javaClass.simpleName) }
    }
    @Synchronized override fun close() {
        active = false
        trackers.values.forEach { runCatching { it.unsetEventListener() } }
        trackers.clear()
        runCatching { service.disconnectService() }
    }
}
