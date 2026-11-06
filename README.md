# Infralibre: Application Inference from Packet Flows

Infralibre was awarded third place in the ITU AI/ML Challenge on Application Inference from Packet Flows. This repository provides a **public-data reference implementation and refitted model**, with a separate account of the historical competition submission in the [technical report](TECHNICAL_REPORT.md).

The archived V9 submission scored **95.007770/100**. The refitted checkpoint reproduces the archived V4 baseline prediction file exactly. The V9 result also reflects historical manual adjustments. A separate audit reproduced V9 byte for byte from the regenerated V4 baseline and the recorded adjustments. The checkpoint alone produces V4 and has no official evaluator score.

## Run the reference

Use Python 3.12. Download the participant-facing `dataset/` directory from the [challenge starter kit](https://github.com/FG-AI4H/application-inference-packet-flows-public). The dataset is not redistributed here. The `--data` argument must point to the directory containing `task1_static/` and `task2_realtime_prefix5/`.

```sh
python -m pip install -r requirements.txt
python run_reference.py --data /path/to/dataset --model public_reference_models.joblib --output reference_predictions.json
```

To refit on the public training and validation labels:

```sh
python run_reference.py --data /path/to/dataset --fit all --model refit.joblib --output refit_predictions.json
```

To score the reference on the released validation set using only the training labels for fitting:

```sh
python run_reference.py --data /path/to/dataset --fit train --split validation --model validation_models.joblib --output validation_predictions.json
```

Add `--prefix-only` to train or run Task 2 with only the prefix-five directory present. The reference uses aggregate class counts published in the challenge metadata. Count assignment acts on a batch; this repository does not benchmark per-flow online inference or latency.

The model artifact uses joblib serialization. Load checkpoints only from a trusted source. `experiment_v4.py` preserves the original model definitions; use `run_reference.py` for the documented workflow.

Author: Bassem Abi-Farah (Canada), Infralibre.
