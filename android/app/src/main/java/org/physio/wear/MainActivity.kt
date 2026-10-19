package org.physio.wear

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.wear.compose.foundation.lazy.ScalingLazyColumn
import androidx.wear.compose.foundation.lazy.ScalingLazyListState
import androidx.wear.compose.material.*
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : ComponentActivity() {
    private lateinit var engine: MonitoringEngine
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        engine = MonitoringEngine(applicationContext)
        setContent { PhysioApp(engine) }
    }
    override fun onPause() { if (::engine.isInitialized) engine.stop(); super.onPause() }
    override fun onDestroy() { engine.close(); super.onDestroy() }
}

private val Ink = Color(0xFFE9E6DF)
private val Accent = Color(0xFFB6D5BF)
private val Muted = Color(0xFFAAA9A1)

@Composable private fun RowButton(
    label: String, secondary: String? = null, enabled: Boolean = true, onClick: () -> Unit
) {
    Chip(
        onClick = onClick, enabled = enabled, modifier = Modifier.fillMaxWidth(),
        colors = ChipDefaults.chipColors(backgroundColor = Color(0xFF232724), contentColor = Ink),
        label = { Text(label) }, secondaryLabel = secondary?.let { { Text(it) } }
    )
}

@Composable private fun BackButton(label: String = "Back", onClick: () -> Unit) {
    val foreground = Color(0xFFD9ECF5)
    Chip(
        onClick = onClick, modifier = Modifier.fillMaxWidth(),
        colors = ChipDefaults.chipColors(
            backgroundColor = Color(0xFF263F4C), contentColor = foreground
        ),
        icon = {
            Canvas(Modifier.size(20.dp)) {
                val width = 2.dp.toPx()
                val tip = Offset(size.width * .2f, size.height * .5f)
                drawLine(foreground, Offset(size.width * .85f, tip.y), tip,
                    strokeWidth = width, cap = StrokeCap.Round)
                drawLine(foreground, tip, Offset(size.width * .5f, size.height * .2f),
                    strokeWidth = width, cap = StrokeCap.Round)
                drawLine(foreground, tip, Offset(size.width * .5f, size.height * .8f),
                    strokeWidth = width, cap = StrokeCap.Round)
            }
        },
        label = { Text(label) }
    )
}

@Composable private fun ReplaySensorOption(
    sensor: String, selected: Boolean, enabled: Boolean, onClick: () -> Unit
) {
    Column(Modifier.fillMaxWidth().padding(start = 16.dp)) {
        Chip(
            onClick = onClick, enabled = enabled, modifier = Modifier.fillMaxWidth(),
            colors = ChipDefaults.chipColors(
                backgroundColor = Color(0xFF1C3025), contentColor = Ink
            ),
            icon = {
                Canvas(Modifier.size(16.dp)) {
                    drawCircle(
                        if (selected) Accent else Muted.copy(alpha = .5f),
                        radius = size.minDimension * .3f
                    )
                }
            },
            label = { Text(sensorLabel(sensor)) },
            secondaryLabel = { Text(if (selected) "On" else "Off") }
        )
    }
}

@Composable private fun Readings(state: MonitorState) {
    Column(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 26.dp, vertical = 6.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        supportingReadings(state).chunked(2).forEach { readings ->
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                readings.forEach { (label, value) ->
                    Column(Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(label, style = MaterialTheme.typography.caption2, color = Muted)
                        Text(value, style = MaterialTheme.typography.body2, color = Ink)
                    }
                }
            }
        }
    }
}

@Composable private fun StressResult(state: MonitorState, engine: MonitoringEngine) {
    val view = LocalView.current
    Column(
        modifier = Modifier.fillMaxWidth().padding(bottom = 6.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("Stress level", style = MaterialTheme.typography.title3)
        Text(
            state.probability?.let { "%.0f%%".format(it * 100) } ?: "--",
            style = MaterialTheme.typography.display1, color = Accent,
            modifier = Modifier.drawWithContent {
                drawContent()
                val id = state.predictionTraceId
                if (engine.profiler.registerDraw(id)) {
                    view.viewTreeObserver.registerFrameCommitCallback {
                        engine.profiler.commitFrame(id, android.os.SystemClock.elapsedRealtimeNanos())
                    }
                }
            }
        )
        Readings(state)
    }
}

@Composable fun PhysioApp(engine: MonitoringEngine, initialScreen: String = "Home") {
    val state by engine.state.collectAsState()
    val context = LocalContext.current
    val preferences = remember {
        context.getSharedPreferences("settings", android.content.Context.MODE_PRIVATE)
    }
    var screen by rememberSaveable {
        mutableStateOf(when (initialScreen) {
            "History", "Settings" -> initialScreen
            "Model" -> "Settings"
            else -> "Home"
        })
    }
    var mode by remember {
        val defaultMode = if (engine.replayAvailable) "Replay" else "Live"
        val savedMode = preferences.getString("mode", defaultMode) ?: defaultMode
        mutableStateOf(if (savedMode == "Replay" && !engine.replayAvailable) "Live" else savedMode)
    }
    var interval by remember { mutableStateOf(preferences.getInt("interval", 10)) }
    var sampleRate by remember { mutableStateOf(preferences.getInt("sample_rate", 32)) }
    var replaySpeed by remember { mutableStateOf(preferences.getInt("replay_speed", 10)) }
    var replaySensors by remember { mutableStateOf(SensorPolicy.all) }
    var showModel by rememberSaveable { mutableStateOf(initialScreen == "Model") }
    var showReplaySensors by rememberSaveable { mutableStateOf(false) }
    var confirmClear by rememberSaveable { mutableStateOf(false) }
    val listState = remember(screen) { ScalingLazyListState(initialCenterItemIndex = 0) }
    BackHandler(enabled = screen != "Home") { screen = "Home" }
    LaunchedEffect(mode, interval, sampleRate, replaySpeed) {
        preferences.edit().putString("mode", mode).putInt("interval", interval)
            .putInt("sample_rate", sampleRate).putInt("replay_speed", replaySpeed).apply()
    }
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {
        engine.start(mode, interval, sampleRate, replaySpeed, replaySensors)
    }
    MaterialTheme(colors = Colors(
        primary = Accent, background = Color(0xFF101411), onBackground = Ink, onPrimary = Color.Black
    )) {
        Scaffold(timeText = { TimeText() }) {
            Column(
                modifier = Modifier.fillMaxSize().padding(top = 30.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                if (screen == "Home") StressResult(state, engine)
                ScalingLazyColumn(
                    state = listState, autoCentering = null,
                    contentPadding = PaddingValues(top = 4.dp, bottom = 32.dp),
                    modifier = Modifier.weight(1f).padding(horizontal = 20.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    if (screen != "Home") {
                        item { Text(screen, style = MaterialTheme.typography.title3,
                            textAlign = TextAlign.Center) }
                        item { BackButton { screen = "Home" } }
                    }
                    when (screen) {
                        "Home" -> {
                            item { Text(homeStatus(state), color = Muted, textAlign = TextAlign.Center) }
                            item {
                                RowButton(if (state.running) "Stop" else "Start") {
                                    if (state.running) engine.stop()
                                    else if (mode == "Live" && context.checkSelfPermission(
                                            Manifest.permission.BODY_SENSORS
                                        ) != PackageManager.PERMISSION_GRANTED) {
                                        permission.launch(Manifest.permission.BODY_SENSORS)
                                    } else engine.start(mode, interval, sampleRate, replaySpeed, replaySensors)
                                }
                            }
                            val recent = state.history.filter {
                                it.mode == state.mode && it.modelId == engine.modelId
                            }
                            if (recent.isNotEmpty()) item { Trend(recent) }
                            item { RowButton("History") { screen = "History" } }
                            item { RowButton("Settings") { screen = "Settings" } }
                        }
                        "History" -> {
                            if (state.history.isEmpty()) {
                                item { Text("No saved readings yet", color = Muted) }
                            } else {
                                state.history.groupBy { Triple(it.sessionId, it.modelId, it.mode) }
                                    .entries.take(5).forEach { (key, readings) ->
                                        item {
                                            Text(
                                                SimpleDateFormat("MMM d · HH:mm", Locale.US)
                                                    .format(Date(readings.first().timestamp)),
                                                style = MaterialTheme.typography.title3
                                            )
                                        }
                                        if (key.third == "Replay") item { Text("Replay", color = Muted) }
                                        item {
                                            Text(
                                                "Average stress  %.0f%%".format(
                                                    readings.map { it.probability }.average() * 100
                                                ),
                                                textAlign = TextAlign.Center
                                            )
                                        }
                                        item { Trend(readings) }
                                    }
                                item {
                                    RowButton(
                                        if (confirmClear) "Confirm clear history" else "Clear history"
                                    ) {
                                        if (confirmClear) { engine.clearHistory(); confirmClear = false }
                                        else confirmClear = true
                                    }
                                }
                            }
                            item { BackButton("Back to home") { screen = "Home" } }
                        }
                        "Settings" -> {
                            if (state.running) {
                                item { Text("Stop monitoring to change settings", color = Muted,
                                    textAlign = TextAlign.Center) }
                            }
                            item {
                                RowButton(
                                    "Data source: " + if (mode == "Live") "Watch" else "Replay",
                                    enabled = !state.running && engine.replayAvailable
                                ) { mode = if (mode == "Replay") "Live" else "Replay" }
                            }
                            item {
                                RowButton("Updates: every ${interval} seconds", enabled = !state.running) {
                                    interval = if (interval == 10) 5 else 10
                                }
                            }
                            if (mode == "Live") {
                                item {
                                    RowButton(
                                        "Sensor detail: " + if (sampleRate == 32) "Standard" else "High",
                                        enabled = !state.running
                                    ) { sampleRate = if (sampleRate == 32) 64 else 32 }
                                }
                            } else {
                                item {
                                    RowButton("Playback speed: ${replaySpeed}x", enabled = !state.running) {
                                        replaySpeed = if (replaySpeed == 10) 1 else 10
                                    }
                                }
                                item {
                                    RowButton(
                                        "Replay sensors",
                                        if (showReplaySensors) "Tap a sensor to change it"
                                        else "${replaySensors.size} on · tap to edit"
                                    ) { showReplaySensors = !showReplaySensors }
                                }
                                if (showReplaySensors) SensorPolicy.all.forEach { sensor ->
                                    item {
                                        ReplaySensorOption(
                                            sensor, selected = sensor in replaySensors, enabled = !state.running
                                        ) {
                                            replaySensors = if (sensor in replaySensors)
                                                replaySensors - sensor else replaySensors + sensor
                                        }
                                    }
                                }
                            }
                            item { RowButton("Model details") { showModel = !showModel } }
                            if (showModel) {
                                modelInformation(engine.metadata).forEach { (label, value) ->
                                    item {
                                        Text("$label\n$value", color = Muted, textAlign = TextAlign.Center)
                                    }
                                }
                            }
                            item { BackButton("Back to home") { screen = "Home" } }
                        }
                    }
                }
            }
        }
    }
}

@Composable private fun Trend(history: List<Estimate>) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text("Recent stress levels", color = Muted)
        Canvas(Modifier.fillMaxWidth().height(56.dp)) {
            drawLine(Muted, Offset(0f, size.height), Offset(size.width, size.height))
            val values = history.take(30).reversed()
            values.forEachIndexed { i, value ->
                val x = if (values.size > 1) i.toFloat() / (values.size - 1) * size.width else size.width / 2
                drawCircle(Accent, radius = 2f, center = Offset(x, (1 - value.probability) * size.height))
            }
            values.zipWithNext().forEachIndexed { i, (a, b) ->
                if (canConnectHistory(a, b)) {
                    drawLine(
                        Accent, Offset(i.toFloat() / (values.size - 1) * size.width,
                            (1 - a.probability) * size.height),
                        Offset((i + 1f) / (values.size - 1) * size.width,
                            (1 - b.probability) * size.height), strokeWidth = 2f
                    )
                }
            }
        }
    }
}
