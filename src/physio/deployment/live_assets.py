"""Package completed watch-rate models, exact filter coefficients and real raw validation windows."""
import hashlib
import json
from pathlib import Path
import shutil

from physio.deployment.checkpoints import load_checkpoint
import numpy as np
import onnxruntime as ort
import pandas as pd
from scipy import signal

from physio.evaluation.figures import plt, save_figure
from physio.processing.dataset import discover, load_recording
from physio.processing.live_features import filter_contract, names_for, ppg_features, temperature_features, eda_features
from physio.processing.sensor_contracts import SENSOR_CASES
from physio.processing.acceleration import magnitude_bins, motion_features

CASES = ("acc_bvp", "acc_eda", "acc_temp", "acc_bvp_eda", "acc_bvp_temp", "acc_eda_temp", "full")


def export_live_assets(assets: Path, output: Path, data_root: Path) -> dict:
    root = Path("results/training/live_sensor_subsets")
    for case in CASES:
        marker = root/case/"validated.json"
        if not marker.exists() or json.loads(marker.read_text()).get("status") != "validated":
            raise ValueError(f"Complete live training before export: {case}")
    frame = pd.read_csv("results/preprocessing/live_2025/features.csv")
    held = frame[frame.subject == "S2"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    models, sessions, summaries = [], {}, []
    folder = assets/"sensor_models"
    folder.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        trained = root/case
        names = names_for(SENSOR_CASES[case])
        manifest = json.loads((trained/"training_manifest.json").read_text())
        audit = json.loads((trained/"S2/splits.json").read_text())
        assert manifest["contract"] == "live" and manifest["feature_names"] == names
        assert "S2" not in audit["outer_train"]
        source = trained/"S2/mlp_supervised_2024.onnx"
        dest = folder/f"live_{case}.onnx"
        shutil.copyfile(source,dest)
        session = ort.InferenceSession(str(dest),options,providers=["CPUExecutionProvider"])
        sessions[case] = session
        converted = session.run(None,{"features":held[names].to_numpy(dtype=np.float32)})[0].ravel()
        fitted = load_checkpoint(trained/"S2/mlp_supervised_2024.joblib")
        reference = fitted.predict(held[names].to_numpy())
        error = float(np.max(np.abs(reference-converted)))
        assert error < 5e-5 and np.array_equal(reference >= .5, converted >= .5)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        scores = pd.read_csv(trained/"fold_metrics.csv")
        mean = scores.groupby("subject").macro_f1.mean()
        metadata = {"case":"live_"+case, "sensors":SENSOR_CASES[case],"feature_names":names,
            "file":f"sensor_models/live_{case}.onnx","model_sha256":digest,
            "model_id":f"live_{case}_s2_seed2024_{digest[:12]}","input_count":len(names),
            "model_family":"mlp","model_bytes":dest.stat().st_size,"window_seconds":60,
            "evaluation":[{"macro_f1_mean":float(mean.mean()),"macro_f1_std":float(mean.std())}],
            "sampling_hz":{"BVP":25,"EDA":1,"TEMP":1},"held_out_subject":"S2",
            "python_onnx_max_error":error,"live_transfer_validated":False}
        models.append(metadata)
        summaries.append({"case":case,"sensors":" + ".join(SENSOR_CASES[case]).replace("BVP","PPG"),**metadata["evaluation"][0],"features":len(names),"subjects":len(mean)})
    assets.mkdir(parents=True,exist_ok=True)
    (assets/"live_models.json").write_text(json.dumps({"schema":1,"models":models,
        "EDA":"1 Hz microsiemens; capability checked on supported hardware","BPM":"Display only; never a raw PPG substitute"},indent=2))
    (assets/"live_filters.json").write_text(json.dumps(filter_contract()))
    recording = load_recording(next(p for p in discover(data_root) if p.stem == "S2"))
    ppg = signal.resample_poly(recording.wrist["BVP"][:,0],25,64)
    temperature = signal.resample_poly(recording.wrist["TEMP"][:,0],1,4)
    eda = signal.resample_poly(recording.wrist["EDA"][:,0],1,4)
    bins = magnitude_bins(recording.wrist["ACC"])
    chosen = pd.concat([group.iloc[np.linspace(0,len(group)-1,5,dtype=int)] for _,group in held.groupby("target")])
    vectors = []
    for row in chosen.itertuples():
        start = int(row.start_seconds)
        raw = ppg[start*25:(start+60)*25]
        temp = temperature[start:start+60]
        motion = bins[start*16:(start+60)*16]
        raw_features = dict(zip(names_for(["ACC","BVP","EDA","TEMP"]),
            np.r_[motion_features(motion),ppg_features(raw),eda_features(eda[start:start+60]),temperature_features(temp)]))
        expected = {}
        for case,session in sessions.items():
            selected = np.array([raw_features[n] for n in names_for(SENSOR_CASES[case])],dtype=np.float32)
            expected["live_"+case] = float(session.run(None,{"features":selected[None]})[0][0,0])
        vectors.append({"subject":"S2","start_seconds":start,"target":int(row.target),
            "ppg":raw.tolist(),"eda":eda[start:start+60].tolist(),"temperature":temp.tolist(),"motion_bins":motion.tolist(),
            "features":raw_features,"expected":expected})
    (assets/"live_parity.json").write_text(json.dumps(vectors))
    output.mkdir(parents=True,exist_ok=True)
    evidence = {"status":"python_validated_android_pending","models":models,"raw_validation_windows":len(vectors),
        "sdk_binary_available":True,"physical_watch_validated":False,
        "training_scope":"WESAD resampled continuously to 25Hz PPG and 1Hz EDA/TEMP; not paired watch validation",
        "alignment":"Bounded linear interpolation between adjacent live samples <=2 periods; no extrapolation/long-gap filling"}
    (output/"live_export.json").write_text(json.dumps(evidence,indent=2))
    baseline = pd.read_csv("results/experiments/wear/distillation_v1/fold_metrics.csv")
    baseline = baseline[baseline.model == "mlp_supervised"].groupby("subject").macro_f1.mean()
    summaries.insert(0, {"case":"acc", "sensors":"ACC", "macro_f1_mean":float(baseline.mean()),
        "macro_f1_std":float(baseline.std()), "features":6,"subjects":len(baseline)})
    summary = pd.DataFrame(summaries)
    summary.to_csv("results/stats/live_sensor_models.csv",index=False)
    fig, axis = plt.subplots(figsize=(3.5,3.5))
    axis.errorbar(range(len(summary)),summary.macro_f1_mean,yerr=summary.macro_f1_std,
                  fmt="o",capsize=3,color="#0072B2")
    axis.set_xticks(range(len(summary)),[row["sensors"].replace(" + ","+") for row in summaries], rotation=55, ha="right", fontsize=6.5)
    axis.set(xlabel="Available live inputs",ylabel="Macro-F1 (mean +/- subject SD)",ylim=(0,1.1))
    axis.set_title("Resampled WESAD; physical-watch accuracy unvalidated",fontsize=6.5)
    save_figure(fig,Path("results/figures/live_sensor_comparison"))
    return evidence
