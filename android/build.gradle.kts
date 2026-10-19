plugins {
    id("com.android.application") version "8.7.0" apply false
    id("org.jetbrains.kotlin.android") version "1.9.25" apply false
}

layout.buildDirectory.set(file("../results/build/android"))
subprojects {
    layout.buildDirectory.set(rootProject.layout.buildDirectory.dir(name))
}
