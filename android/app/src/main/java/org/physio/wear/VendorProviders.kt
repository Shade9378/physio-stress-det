package org.physio.wear

import android.content.Context
import android.os.Build

interface VendorProvider {
    fun create(context: Context, onSample: (PhysiologicalSample) -> Unit, onStatus: (String) -> Unit): VendorSensorSource
}

object VendorProviders {
    fun create(context: Context, onSample: (PhysiologicalSample) -> Unit,
        onStatus: (String) -> Unit): List<VendorSensorSource> {
        if (!Build.MANUFACTURER.equals("samsung", ignoreCase = true)) {
            onStatus("Standard Wear OS APIs: raw PPG/EDA/skin TEMP not exposed by configured APIs")
            return emptyList()
        }
        return try {
            val provider = Class.forName("org.physio.wear.SamsungProvider").getDeclaredConstructor().newInstance() as VendorProvider
            listOf(provider.create(context,onSample,onStatus))
        } catch (error: ClassNotFoundException) {
            onStatus("Samsung SDK 1.4.1 not packaged · PPG/EDA/skin TEMP unavailable")
            emptyList()
        } catch (error: Exception) {
            onStatus("Samsung provider unavailable: " + error.javaClass.simpleName)
            emptyList()
        }
    }
}
