package org.physio.wear

import android.content.Context
import org.json.JSONObject
import kotlin.math.*

/** Python physio.processing.live_features contract: raw green PPG 25 Hz and contact skin TEMP 1 Hz. */
class LiveFeatures(context: Context) {
    private val filters = JSONObject(context.assets.open("generated/live_filters.json").bufferedReader().use { it.readText() })
    private fun filter(values: DoubleArray, name: String): DoubleArray {
        val definition = filters.getJSONObject(name)
        val sos = definition.getJSONArray("sos").let { a -> (0 until a.length()).map { i ->
            a.getJSONArray(i).let { b -> DoubleArray(b.length()) { b.getDouble(it) } }
        } }
        val zi = definition.getJSONArray("zi")
        val pad = definition.getInt("pad")
        require(values.size > pad && values.all { it.isFinite() })
        val extended = DoubleArray(values.size + 2*pad) { i ->
            when {
                i < pad -> 2*values.first()-values[pad-i]
                i >= pad+values.size -> 2*values.last()-values[2*values.size+pad-2-i]
                else -> values[i-pad]
            }
        }
        fun forward(input: DoubleArray): DoubleArray {
            val scale = input.first()
            var output = input.copyOf()
            for (section in sos.indices) {
                val c = sos[section]
                var z1 = zi.getJSONArray(section).getDouble(0)*scale
                var z2 = zi.getJSONArray(section).getDouble(1)*scale
                output = DoubleArray(output.size) { i ->
                    val x = output[i]
                    val y = c[0]*x+z1
                    z1 = c[1]*x-c[4]*y+z2
                    z2 = c[2]*x-c[5]*y
                    y
                }
            }
            return output
        }
        return forward(forward(extended).reversedArray()).reversedArray().copyOfRange(pad, pad+values.size)
    }
    private fun std(x: DoubleArray): Double {
        val mean = x.average()
        return sqrt(x.sumOf { (it-mean).pow(2) }/x.size)
    }
    private fun quantile(values: DoubleArray, q: Double): Double {
        val x = values.sortedArray()
        val i = (x.size-1)*q
        val low = floor(i).toInt()
        val high = ceil(i).toInt()
        return x[low]+(x[high]-x[low])*(i-low)
    }
    fun ppg(raw: DoubleArray): Map<String, Double> {
        require(raw.size == 1500)
        val filtered = filter(raw, "PPG")
        val mean = filtered.average()
        val scale = std(filtered)
        require(scale >= 1e-9) { "Flat raw PPG" }
        val x = DoubleArray(filtered.size) { (filtered[it]-mean)/scale }
        val candidates = mutableListOf<Int>()
        var i = 1
        while (i < x.lastIndex) {
            if (x[i-1] < x[i]) {
                var right = i
                while (right < x.lastIndex && x[right+1] == x[i]) right++
                if (right < x.lastIndex && x[right+1] < x[i]) candidates.add((i+right)/2)
                i = right
            }
            i++
        }
        val kept = mutableListOf<Int>()
        for (peak in candidates.sortedWith(compareByDescending<Int> { x[it] }.thenByDescending { it })) {
            if (kept.all { abs(it-peak) >= 8 }) kept.add(peak)
        }
        val peaks = kept.sorted().filter { peak ->
            var left = x[peak]; var right = x[peak]; var k = peak
            while (k >= 0 && x[k] <= x[peak]) { left = min(left, x[k]); k-- }
            k = peak
            while (k < x.size && x[k] <= x[peak]) { right = min(right, x[k]); k++ }
            x[peak]-max(left,right) >= .3
        }
        val intervals = peaks.zipWithNext().map { (a,b) -> (b-a)/25.0 }.filter { it >= .3 && it <= 2.0 }.toDoubleArray()
        require(intervals.size >= 3) { "Insufficient raw PPG pulse intervals" }
        val delta = intervals.toList().zipWithNext().map { (a,b) -> (b-a).pow(2) }
        return linkedMapOf("bvp_bpm" to 60/quantile(intervals,.5),
            "bvp_rmssd_seconds" to sqrt(delta.average()), "bvp_interval_std_seconds" to std(intervals),
            "bvp_valid_intervals" to intervals.size.toDouble(),
            "bvp_normalized_iqr" to quantile(x,.75)-quantile(x,.25))
    }
    fun temperature(raw: DoubleArray): Map<String, Double> = scalar(raw,"TEMP","temp")
    fun eda(raw: DoubleArray): Map<String, Double> {
        require(raw.all { it >= 0 && it.isFinite() })
        return scalar(raw,"EDA","eda")
    }
    private fun scalar(raw: DoubleArray, sensor: String, prefix: String): Map<String, Double> {
        require(raw.size == 60)
        val x = filter(raw,sensor)
        val center = (x.size-1)/2.0
        val slope = x.indices.sumOf { (it-center)*x[it] }/x.indices.sumOf { (it-center).pow(2) }
        val delta = DoubleArray(x.size-1) { x[it+1]-x[it] }
        return linkedMapOf("${prefix}_mean" to x.average(), "${prefix}_std" to std(x), "${prefix}_min" to x.min(),
            "${prefix}_max" to x.max(), "${prefix}_iqr" to quantile(x,.75)-quantile(x,.25),
            "${prefix}_slope" to slope, "${prefix}_diff_std" to std(delta))
    }
}
