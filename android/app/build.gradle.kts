import java.security.MessageDigest
import java.util.zip.ZipFile
import java.util.jar.JarInputStream

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}
android {
    sourceSets.getByName("main").assets.srcDir("../../results/app/assets")
    namespace = "org.physio.wear"
    compileSdk = 34
    buildToolsVersion = "34.0.0"
    defaultConfig {
        applicationId = "org.physio.wear"
        minSdk = 30
        targetSdk = 34
        versionCode = 3
        versionName = "0.2.1"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    splits {
        abi {
            isEnable = true
            reset()
            include("armeabi-v7a", "arm64-v8a", "x86_64")
            isUniversalApk = false
        }
    }
    buildFeatures { compose = true }
    composeOptions { kotlinCompilerExtensionVersion = "1.5.15" }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    packaging { resources.excludes += "/META-INF/{AL2.0,LGPL2.1}" }
    testOptions { unitTests.isReturnDefaultValues = true }
}
dependencyLocking { lockAllConfigurations() }
dependencies {
    implementation("androidx.activity:activity-compose:1.9.2")
    implementation("androidx.compose.ui:ui:1.7.5")
    implementation("androidx.compose.foundation:foundation:1.7.5")
    implementation("androidx.wear.compose:compose-material:1.4.0")
    implementation("androidx.wear.compose:compose-foundation:1.4.0")
    implementation("com.google.guava:guava:31.1-android")
    implementation("androidx.health:health-services-client:1.0.0-rc02")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
}

// Optional vendor library with a pinned version and checksum.
val bundledSamsung = rootProject.file("../results/tooling/samsung/1.4.1/samsung-health-sensor-api-1.4.1.aar")
val samsungSdk = providers.gradleProperty("samsungSdk").orNull
    ?: bundledSamsung.takeIf { it.exists() }?.absolutePath
if (samsungSdk != null) {
    val archive = file(samsungSdk)
    require(archive.name == "samsung-health-sensor-api-1.4.1.aar") { "Require the pinned SDK 1.4.1 AAR" }
    val expected = "893cd5d6564db0f304bf511a555c1d65ca6bccc8475fc979ff1d71d50680344c"
    providers.gradleProperty("samsungSdkSha256").orNull?.let { require(it == expected) }
    val actual = MessageDigest.getInstance("SHA-256").digest(archive.readBytes()).joinToString("") { "%02x".format(it) }
    require(actual == expected) { "Samsung SDK checksum mismatch" }
    val symbols = mutableMapOf<String,ByteArray>()
    ZipFile(archive).use { aar ->
        JarInputStream(aar.getInputStream(aar.getEntry("classes.jar"))).use { jar ->
            var entry = jar.nextJarEntry
            while (entry != null) {
                symbols[entry.name] = jar.readBytes()
                entry = jar.nextJarEntry
            }
        }
    }
    val tracker = symbols["com/samsung/android/service/health/tracking/data/HealthTrackerType.class"]
        ?.toString(Charsets.ISO_8859_1) ?: error("Invalid Samsung SDK")
    require("PPG_CONTINUOUS" in tracker && "SKIN_TEMPERATURE_CONTINUOUS" in tracker &&
        "EDA_CONTINUOUS" in tracker) { "SDK symbols do not match the pinned 2025 contract" }
    require(symbols.containsKey("com/samsung/android/service/health/tracking/data/ValueKey${'$'}PpgSet.class"))
    android.sourceSets.getByName("main").java.srcDir("src/samsung/java")
    dependencies.add("implementation", files(archive))
}

val verifyRuntimeAssets by tasks.registering {
    val assets = file("../../results/app/assets/generated")
    val required = listOf(
        "model.json", "stress.onnx", "sensor_models.json", "live_models.json", "live_filters.json",
        "sensor_models/acc.onnx", "sensor_models/acc_bvp.onnx", "sensor_models/acc_eda.onnx",
        "sensor_models/acc_temp.onnx", "sensor_models/acc_bvp_eda.onnx",
        "sensor_models/acc_bvp_temp.onnx", "sensor_models/acc_eda_temp.onnx",
        "sensor_models/full.onnx", "sensor_models/no_acc.onnx",
        "sensor_models/live_acc_bvp.onnx", "sensor_models/live_acc_eda.onnx",
        "sensor_models/live_acc_temp.onnx", "sensor_models/live_acc_bvp_eda.onnx",
        "sensor_models/live_acc_bvp_temp.onnx", "sensor_models/live_acc_eda_temp.onnx",
        "sensor_models/live_full.onnx"
    )
    doLast {
        val missing = required.filter { !assets.resolve(it).isFile }
        check(missing.isEmpty()) {
            "Runtime model assets missing: " + missing.joinToString() +
                ". Download and install the model release with scripts/model_assets.py; see README."
        }
    }
}
tasks.named("preBuild").configure { dependsOn(verifyRuntimeAssets) }
