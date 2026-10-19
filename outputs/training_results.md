# Training results

[Technical details](technical_details.md) | [Setup](../README.md#build-and-run)

## Training setup

| Setting | Value |
|---|---|
| Dataset | WESAD, 15 subjects; local recordings only |
| Labels | Baseline = 0, stress = 1; other conditions and mixed windows excluded |
| Windows | 60 seconds, 10-second stride; 2,575 windows (1,671 baseline, 904 stress) |
| Evaluation | Leave-one-subject-out; 14 training subjects, one held out |
| Leakage prevention | Subject-disjoint tuning; normalization fitted on training subjects only |
| MLP | One 16-unit ReLU hidden layer; sigmoid output |
| MLP optimizer / loss | Adam; balanced weighted binary cross-entropy |
| MLP learning rate / weight decay | 0.01 / 0.001 |
| MLP batch / epoch selection | Full batch; 60 or 160 epochs, selected by three-fold subject-grouped inner validation |
| MLP seeds / device | 2024, 17, 42; FP32 CPU, two threads |
| Final fit | Refit on all 14 training subjects using the selected epoch count

## Original model comparison

| Model | Accuracy | Macro-F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic regression | 0.929 +/- 0.106 | 0.924 +/- 0.109 | 0.942 +/- 0.079 | 0.928 +/- 0.091 |
| Random forest | 0.921 +/- 0.098 | 0.899 +/- 0.154 | 0.906 +/- 0.171 | 0.906 +/- 0.127 |
| XGBoost | 0.912 +/- 0.108 | 0.897 +/- 0.134 | 0.933 +/- 0.078 | 0.903 +/- 0.116 |
| 1D CNN | 0.667 +/- 0.248 | 0.603 +/- 0.277 | 0.727 +/- 0.261 | 0.674 +/- 0.217 |
| TCN | 0.647 +/- 0.257 | 0.574 +/- 0.297 | 0.574 +/- 0.335 | 0.650 +/- 0.243 |
| Sensor-specific encoders | 0.775 +/- 0.189 | 0.733 +/- 0.221 | 0.815 +/- 0.182 | 0.765 +/- 0.175 |
| MLP | 0.933 +/- 0.091 | 0.928 +/- 0.094 | 0.942 +/- 0.076 | 0.933 +/- 0.082 |

Classical models and the MLP use the original 48-feature table. Sequence models use six-channel signals.

Classical tuning uses three-fold grouped validation. Sequence models use one training subject for epoch selection, then refit on 14 subjects: Adam, learning rate 0.001, batch 64, up to 12 epochs. Models other than the MLP use seed 2024.

These are not the native live-model scores.

## Available-sensor models

MLPs trained on each live input combination:

| Available sensors | Macro-F1 |
|---|---|
| ACC | 0.7837 +/- 0.1458 |
| ACC + PPG | 0.8294 +/- 0.1860 |
| ACC + EDA | 0.8052 +/- 0.1822 |
| ACC + TEMP | 0.8155 +/- 0.2087 |
| ACC + PPG + EDA | 0.8092 +/- 0.1647 |
| ACC + PPG + TEMP | 0.9074 +/- 0.1510 |
| ACC + EDA + TEMP | 0.8917 +/- 0.0994 |
| ACC + PPG + EDA + TEMP | 0.9074 +/- 0.1376 |

All four live sensors produce 25 features. Seven additional combinations completed 315 LOSO fits; the ACC-only baseline reuses its completed three-seed experiment. More sensors did not consistently improve performance.

The separate 32-feature prepared-replay MLP achieved **0.9203 +/- 0.0898** macro-F1.

## Inference optimization

ACC-only, six-feature LR and MLP; 15 LOSO folds, seed 2024:

| Model | Execution format | Accuracy | Macro-F1 | Changed labels |
|---|---|---|---|---|
| LR | PyTorch FP32 | 0.749 +/- 0.110 | 0.706 +/- 0.134 | 0 |
| LR | ONNX FP32 | 0.749 +/- 0.110 | 0.706 +/- 0.134 | 0 |
| LR | ONNX mixed FP16 | 0.749 +/- 0.110 | 0.706 +/- 0.134 | 0 |
| LR | ONNX dynamic INT8 | 0.749 +/- 0.112 | 0.706 +/- 0.136 | 23 |
| MLP | PyTorch FP32 | 0.809 +/- 0.148 | 0.789 +/- 0.151 | 0 |
| MLP | ONNX FP32 | 0.809 +/- 0.148 | 0.789 +/- 0.151 | 0 |
| MLP | ONNX mixed FP16 | 0.809 +/- 0.148 | 0.789 +/- 0.151 | 0 |
| MLP | ONNX dynamic INT8 | 0.808 +/- 0.149 | 0.788 +/- 0.151 | 17 |

Latency and RAM were measured on the computer CPU, including process/runtime overhead for RAM.

## App response profile

| Sequential stage | LR: mean / p95 (ms) | MLP: mean / p95 (ms) |
|---|---|---|
| 1. Convert sample + complete window | 1.143 / 2.463 | 0.801 / 1.946 |
| 2. Extract six movement features | 0.408 / 0.918 | 0.390 / 0.587 |
| 3. Run model | 1.134 / 2.507 | 1.455 / 7.971 |
| 4. Save + read local history | 13.611 / 26.096 | 13.492 / 19.975 |
| 5. Prepare state publication | 0.014 / 0.020 | 0.031 / 0.118 |
| 6. Publish state + commit UI frame | 49.254 / 89.787 | 63.426 / 221.211 |
| Total: last sample to frame commit | 65.565 / 115.375 | 79.595 / 244.426 |

Wear OS x86_64 emulator, ACC-only LR and MLP, three replay sessions and 48 predictions per model. End-to-end runs from the last sample through preprocessing, inference, storage and UI frame commit, excludes the initial 60-second collection and model loading.

**Note:** Benchmarks are run on an emulator, not a physical watch.
