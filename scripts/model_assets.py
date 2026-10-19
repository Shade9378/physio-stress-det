"""Package and install watch model assets without datasets or vendor SDK binaries.

Uses only the Python 3.11 standard library; run from any working directory.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS = PROJECT_ROOT / "results/app/assets/generated"
DEFAULT_RELEASE = PROJECT_ROOT / "results/releases"
ARCHIVE_NAME = "physio-model-assets-v1.zip"
MAX_BUNDLE_BYTES = 5 * 1024 * 1024
CONTRACT_FILES = {"model.json", "sensor_models.json", "live_models.json", "live_filters.json"}
LIVE_CASES = {
    "live_acc_bvp", "live_acc_eda", "live_acc_temp", "live_acc_bvp_eda",
    "live_acc_bvp_temp", "live_acc_eda_temp", "live_full",
}
REPLAY_CASES = {
    "acc", "acc_bvp", "acc_eda", "acc_temp", "acc_bvp_eda",
    "acc_bvp_temp", "acc_eda_temp", "full", "no_acc",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def model_entries(payload: dict[str, bytes]) -> list[dict]:
    entries = []
    for filename, cases in [
        ("sensor_models.json", REPLAY_CASES), ("live_models.json", LIVE_CASES)
    ]:
        registry = json.loads(payload[filename])
        if registry["schema"] != 1:
            raise ValueError(f"Unsupported registry: {filename}")
        models = registry["models"]
        if {model["case"] for model in models} != cases or len(models) != len(cases):
            raise ValueError(f"Incomplete or duplicate model cases: {filename}")
        entries.extend(models)
    return entries


def required_files(payload: dict[str, bytes]) -> set[str]:
    names = CONTRACT_FILES | {"stress.onnx"}
    for model in model_entries(payload):
        name = model["file"]
        if not re.fullmatch(r"sensor_models/[a-z_]+\.onnx", name):
            raise ValueError(f"Unsafe model path: {name}")
        names.add(name)
    return names


def verify_payload(payload: dict[str, bytes]) -> None:
    if set(payload) != required_files(payload):
        raise ValueError("Bundle contains missing or unexpected files")
    baseline = json.loads(payload["model.json"])
    if baseline["schema"] != 2 or baseline["model_family"] != "mlp":
        raise ValueError("Unsupported baseline metadata")
    if len(baseline["feature_names"]) != 6 or baseline["window_seconds"] != 60:
        raise ValueError("Unsupported baseline input contract")
    if sha256(payload["stress.onnx"]) != baseline["model_sha256"]:
        raise ValueError("Baseline model checksum mismatch")
    json.loads(payload["live_filters.json"])
    for model in model_entries(payload):
        if model["model_family"] != "mlp" or model["input_count"] != len(model["feature_names"]):
            raise ValueError(f"Invalid model contract: {model['case']}")
        if sha256(payload[model["file"]]) != model["model_sha256"]:
            raise ValueError(f"Model checksum mismatch: {model['case']}")


def read_assets(assets_dir: Path) -> dict[str, bytes]:
    payload = {name: (assets_dir / name).read_bytes() for name in CONTRACT_FILES}
    payload.update({name: (assets_dir / name).read_bytes() for name in required_files(payload)})
    verify_payload(payload)
    return payload


def write_archive(assets_dir: Path, output_dir: Path) -> Path:
    payload = read_assets(assets_dir)
    manifest = {
        "schema": 1,
        "bundle": "physio-watch-models",
        "version": 1,
        "includes_replay_data": False,
        "includes_vendor_sdk": False,
        "files": {
            "generated/" + name: {"sha256": sha256(data), "bytes": len(data)}
            for name, data in sorted(payload.items())
        },
    }
    files = {"manifest.json": (json.dumps(manifest, indent=2) + "\n").encode()}
    files.update({"generated/" + name: data for name, data in payload.items()})
    # Fixed ZIP metadata makes repeated packages byte-for-byte reproducible.
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    data = stream.getvalue()
    if len(data) > MAX_BUNDLE_BYTES:
        raise ValueError("Model archive exceeds the release size limit")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / ARCHIVE_NAME
    output.write_bytes(data)
    (output_dir / "SHA256SUMS").write_text(f"{sha256(data)}  {ARCHIVE_NAME}\n", encoding="utf-8")
    read_archive(output, output_dir / "SHA256SUMS")
    return output


def read_archive(archive_path: Path, checksum_path: Path) -> dict[str, bytes]:
    if archive_path.stat().st_size > MAX_BUNDLE_BYTES:
        raise ValueError("Model archive exceeds the release size limit")
    checksums = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest) or name in checksums:
            raise ValueError("Invalid checksum file")
        checksums[name] = digest.lower()
    data = archive_path.read_bytes()
    if checksums.get(archive_path.name) != sha256(data):
        raise ValueError("Release archive checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) > 100 or len(names) != len(set(names)):
            raise ValueError("Too many or duplicate ZIP entries")
        if sum(entry.file_size for entry in entries) > MAX_BUNDLE_BYTES:
            raise ValueError("Uncompressed model archive exceeds the size limit")
        for name in names:
            path = PurePosixPath(name)
            if "\\" in name or path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe archive path")
        manifest = json.loads(archive.read("manifest.json"))
        if (manifest["schema"] != 1 or manifest["bundle"] != "physio-watch-models"
                or manifest["version"] != 1 or manifest["includes_replay_data"]
                or manifest["includes_vendor_sdk"]):
            raise ValueError("Unsupported model bundle")
        expected = manifest["files"]
        if set(names) != set(expected) | {"manifest.json"}:
            raise ValueError("Archive contents do not match manifest")
        payload = {}
        for name, item in expected.items():
            if not name.startswith("generated/"):
                raise ValueError("Unexpected bundle destination")
            content = archive.read(name)
            if len(content) != item["bytes"] or sha256(content) != item["sha256"]:
                raise ValueError(f"File checksum mismatch: {name}")
            payload[name.removeprefix("generated/")] = content
    verify_payload(payload)
    return payload


def install_archive(
    archive_path: Path, checksum_path: Path, assets_dir: Path, replace: bool = False
) -> int:
    payload = read_archive(archive_path, checksum_path)
    assets_dir = assets_dir.resolve()
    # Validate every existing destination before writing anything.
    for name, data in payload.items():
        destination = assets_dir / name
        if not destination.resolve().is_relative_to(assets_dir):
            raise ValueError("Destination escapes the asset directory")
        if destination.exists() and destination.read_bytes() != data and not replace:
            raise ValueError(f"Different asset exists: {name}; use --replace to update models")
    assets_dir.mkdir(parents=True, exist_ok=True)
    for name, data in payload.items():
        destination = assets_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            temporary.write(data)
            temporary_path = Path(temporary.name)
        try:
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)
    verify_payload(read_assets(assets_dir))
    return len(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("package", help="Prepare the two files to attach to a GitHub Release")
    package.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS)
    package.add_argument("--output-dir", type=Path, default=DEFAULT_RELEASE)
    for command in ("install", "verify"):
        operation = commands.add_parser(command, help=f"{command.title()} a downloaded release archive")
        operation.add_argument("--archive", required=True, type=Path)
        operation.add_argument("--checksum", required=True, type=Path)
        if command == "install":
            operation.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS)
            operation.add_argument("--replace", action="store_true")
    arguments = parser.parse_args()
    try:
        if arguments.command == "package":
            output = write_archive(arguments.assets_dir, arguments.output_dir)
            print(f"Ready to upload: {output}\nChecksum file: {output.parent / 'SHA256SUMS'}")
        elif arguments.command == "install":
            count = install_archive(
                arguments.archive, arguments.checksum, arguments.assets_dir, arguments.replace
            )
            print(f"Installed {count} runtime files into {arguments.assets_dir}")
        else:
            payload = read_archive(arguments.archive, arguments.checksum)
            print(f"Verified {len(payload)} runtime files; no replay data or vendor SDK")
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Model assets: {error}\n")


if __name__ == "__main__":
    main()
