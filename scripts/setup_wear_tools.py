"""Download official, checksum-verified pinned Android build tools locally."""
import argparse
from pathlib import Path
import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
import sys

parser = argparse.ArgumentParser(description="Checksum-verified Windows SDK archives for native Wear builds")
parser.add_argument("--samsung-archive", type=Path, help="Install the supplied Samsung SDK 1.4.1 ZIP without downloading tools")
parser.add_argument("--tools-root", type=Path, default=Path("results/tooling/android"))
parser.add_argument("--skip-extract", action="store_true", help="Verify/download archives without extracting again")
arguments = parser.parse_args()
if arguments.samsung_archive:
    destination = Path("results/tooling/samsung/1.4.1")
    with zipfile.ZipFile(arguments.samsung_archive) as archive:
        data = archive.read("1.4.1/libs/samsung-health-sensor-api-1.4.1.aar")
    expected = "893cd5d6564db0f304bf511a555c1d65ca6bccc8475fc979ff1d71d50680344c"
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Samsung SDK checksum mismatch; require official version 1.4.1")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "samsung-health-sensor-api-1.4.1.aar").write_bytes(data)
    print(f"Verified Samsung SDK 1.4.1: {destination}")
    sys.exit(0)
CACHE = arguments.tools_root.resolve()
CACHE.mkdir(parents=True, exist_ok=True)
REPOS = {
    "sdk": "https://dl.google.com/android/repository/repository2-1.xml",
    "wear": "https://dl.google.com/android/repository/sys-img/android-wear/sys-img2-1.xml",
}
records = []
for repo_name, url in REPOS.items():
    tree = ET.fromstring(urllib.request.urlopen(url).read())
    for package in tree.findall("remotePackage"):
        wanted = {"platforms;android-34": "platform-34-ext7_r03.zip",
                  "build-tools;34.0.0": "build-tools_r34-windows.zip",
                  "system-images;android-33;android-wear;x86_64": "x86_64-33_r04.zip"}
        package_id = package.attrib["path"]
        if package_id not in wanted:
            continue
        for archive in package.findall("./archives/archive"):
            complete = archive.find("complete")
            name = complete.findtext("url")
            if name != wanted[package_id]:
                continue
            base = url.rsplit("/", 1)[0] + "/"
            target = CACHE / name
            expected = complete.findtext("checksum")
            print(f"Downloading {package_id}: {name}", flush=True)
            if not target.exists():
                urllib.request.urlretrieve(base + name, target)
            actual = hashlib.sha1(target.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Checksum mismatch {name}")
            with zipfile.ZipFile(target) as bundle:
                if repo_name == "wear":
                    metadata = [n for n in bundle.namelist() if n.endswith("build.prop")]
                    for entry in metadata:
                        (CACHE / "wear_build.prop").write_bytes(bundle.read(entry))
                    dest = CACHE / "sdk/system-images/android-33/android-wear"
                elif package_id.startswith("platforms"):
                    dest = CACHE / "sdk/platforms"
                else:
                    dest = CACHE / "sdk/build-tools"
                dest.mkdir(parents=True, exist_ok=True)
                if not arguments.skip_extract:
                    bundle.extractall(dest)
            if repo_name == "wear":
                properties = (CACHE / "wear_build.prop").read_text()
                timestamp = int(next(line.split("=", 1)[1] for line in properties.splitlines()
                                     if line.startswith("ro.build.date.utc=")))
                assert timestamp <= 1735689599, "System image is newer than the configured release limit"
            if package_id == "build-tools;34.0.0":
                old, new = CACHE / "sdk/build-tools/android-14", CACHE / "sdk/build-tools/34.0.0"
                assert old.resolve().is_relative_to(CACHE) and new.resolve().is_relative_to(CACHE)
                if old.exists() and not new.exists():
                    old.rename(new)
            records.append({"package": package_id, "url": base + name, "sha1": actual})
            print(f"Verified {name}" if arguments.skip_extract else f"Extracted {name}", flush=True)
gradle = CACHE / "gradle-8.9-bin.zip"
url = "https://services.gradle.org/distributions/gradle-8.9-bin.zip"
wrapper = Path("android/gradle/wrapper/gradle-wrapper.properties").read_text()
expected = next(line.split("=", 1)[1] for line in wrapper.splitlines()
                if line.startswith("distributionSha256Sum="))
if not gradle.exists():
    urllib.request.urlretrieve(url, gradle)
if hashlib.sha256(gradle.read_bytes()).hexdigest() != expected:
    raise ValueError("Gradle checksum mismatch")
with zipfile.ZipFile(gradle) as bundle:
    if not arguments.skip_extract:
        bundle.extractall(CACHE)
records.append({"package": "gradle:8.9", "url": url, "sha256": expected})
platform_url = "https://dl.google.com/android/repository/platform-tools_r35.0.2-win.zip"
platform = CACHE / "platform-tools_r35.0.2-win.zip"
if not platform.exists():
    urllib.request.urlretrieve(platform_url, platform)
platform_hash = hashlib.sha256(platform.read_bytes()).hexdigest()
assert platform_hash == "2975a3eac0b19182748d64195375ad056986561d994fffbdc64332a516300bb9"
with zipfile.ZipFile(platform) as bundle:
    if not arguments.skip_extract:
        bundle.extractall(CACHE / "sdk")
records.append({"package": "platform-tools:35.0.2", "url": platform_url, "sha256": platform_hash})
required = {"platforms;android-34", "build-tools;34.0.0", "system-images;android-33;android-wear;x86_64", "gradle:8.9", "platform-tools:35.0.2"}
assert {record["package"] for record in records} == required, "An official pinned archive is unavailable"
(CACHE / "downloads.json").write_text(json.dumps(records, indent=2))
print("DONE", flush=True)
