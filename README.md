# Camera-Motion Compensation for DeepSORT Person Tracking

Omar Hussain and Zayed Mohammed, MAI502, Abu Dhabi University. Everything used for the paper is in this folder.

## Folder map

| Folder | What is inside | Use it to |
|---|---|---|
| `paper/` | Final paper (PDF), LaTeX source, references, figure | read the paper |
| `report/` | `Research_Process_Report.docx` (how the work developed), `Experiment_Results.xlsx` (every number, one sheet per experiment), `supplementary_material.pdf` (tables S1-S11) | see all results in one place |
| `samples/` | 12 short videos, baseline vs CMC side by side, one per experiment (labelled real or simulated) | see the effect visually |
| `code/` | All experiment code, DeepSORT, TrackEval and the models | re-run the experiments (see below) |
| `results/` | Result tables (CSV/JSON) produced by the code | check the numbers behind the paper |
| `protocols/` | Plans written before the added experiments were run | check what was fixed in advance |
| `data/` | Cached inputs (detections, camera-motion estimates), tracker outputs, bootstrap samples, source tables, recorded protocols, the detailed response to the reviewer | re-score without re-running detection |

The videos themselves are not included (MOT17 is 5.9 GB); download them from motchallenge.net (MOT17, MOT15) and huggingface.co/datasets/MCG-NJU/SportsMOT.

## Which code runs which experiment (in `code/`)

| Experiment | Run | Score |
|---|---|---|
| Main study: fixed vs moving MOT17, synthetic pans/shakes | `python run.py all`, `python run.py alt` | `python evaluate.py` |
| Frame skipping | `python run.py stride` | `python evaluate.py` |
| Time-aware Kalman | `python dtaware.py <k>` (k = 2, 3, 6) | `python eval_dt.py` |
| Second detector (MOT17 FRCNN boxes) | `DET=frcnn WORK=work_frcnn python step_rev3.py all` | `python evaluate.py` |
| Independent MOT15 test | `WORK=work_mot15 SEQ_ROOT=<MOT15 train> python mot15.py detect`, then `track` | `python eval_mot15.py` |
| SportsMOT real sweeping camera | `WORK=work_sports SEQ_ROOT=<SportsMOT clips> python sports.py detect`, then `track` | `python eval_sports.py` |
| Virtual PTZ, pole sway, PTZ angle readings | `WORK=work_surv python surv.py detect`, then `track` | `python eval_surv.py` |
| Virtual PTZ with FRCNN boxes / on MOT15 scenes | `python surv2.py detect`, then `track` (settings in the file header) | `python eval_surv2.py` |
| Simulated PPE | `python ppe_run.py`, `python ppe_ci.py` | |
| Confidence intervals | `WORK=... GROUP=full python bootstrap.py 1000` | `python boot_ci4.py` |
| Processing time | `python timing.py` | |

Several drivers stop after about 150 s and continue when run again (our machine limited command length); set `BUDGET=100000` to run them in one go. The appearance model is already converted (`models/mars-small128-cv*.pb`), so the `tools_fix_reid.py` step below can be skipped.

---

# Camera-motion compensation for DeepSORT: experiment code

Runs on CPU with Python 3, OpenCV (>= 4.8 with DNN), numpy, scipy, pandas, matplotlib, protobuf.

## Third-party code and models (fetched, not included)
| Item | Source | Pin |
|---|---|---|
| DeepSORT (GPL-3.0) | github.com/nwojke/deep_sort -> `deep_sort/` | commit f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab |
| TrackEval (MIT) | github.com/JonathonLuiten/TrackEval -> `TrackEval/` | commit 12c8791b303e0a0b50f753af204249e622d0281a + `trackeval_numpy2.patch` (np.float/np.int/np.bool aliases removed in NumPy 2) |
| DeepSORT appearance model `mars-small128.pb` | committed copy in github.com/theAIGuysCode/yolov4-deepsort (`model_data/`) | sha256 e96f3cc0...4a75f1 |
| YOLOv8n COCO, ONNX export | committed copy in github.com/Hyuto/yolov8-onnxruntime-web (`public/model/`) | sha256 505648ad...e00579 |

`python tools_fix_reid.py` then `python -c "import tools_fix_reid as t; t.truncate('models/mars-small128-cv.pb','models/mars-small128-cv-fc1.pb')"`
rewrites the appearance graph so OpenCV can load it (drops the TF map_fn that only reverses channel order; weights unchanged). `reid_ref.py` is a numpy interpreter used to check it.

## Settings
Detector confidence 0.25, NMS IoU 0.8 (capstone inference settings); DeepSORT max cosine 0.2, budget 100, max IoU distance 0.7, max age 30, n_init 3.

## Data
MOT17 training set (CC BY-NC-SA 3.0) under `data/MOT17/train/<seq>/` (one -FRCNN copy per sequence, folder renamed to `MOT17-XX`).

## Run
```
python cmc.py          # self-check
python synth.py        # self-check
python run.py all      # detection, CMC estimation (5 seeds), tracking: base / cmc / oracle
python run.py stride   # frame-subsampling conditions (every 2nd, 3rd, 6th frame)
python evaluate.py     # TrackEval -> work/results.csv
python merges.py       # identity-merge check -> work/paper/merges.csv
python timing.py       # -> work/timing.json
python analysis.py     # tables, figure, summary.json -> work/paper/
```

## Outputs used in the paper
`../results/`: results.csv (TrackEval, every tracker x sequence/group), timing.json, summary.json, merges.csv.
