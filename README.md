# ATML PA3: Fine-Tuning, Vision-Language Models, and Test-Time Compute

Code, configurations, split and example IDs, machine-readable results and figures for Programming Assignment 3 of
EE-5102/CS-6304 Advanced Topics in Machine Learning (Fall 2026). Task 1 fine-tunes Qwen3-0.6B on MetaMathQA with full
fine-tuning, full fine-tuning with L2-SP, LoRA (r = 4, 16) and OPLoRA, and measures what each method gains on GSM8K and
loses on HellaSwag, BoolQ and ARC-Easy. Task 2 connects a frozen SigLIP encoder to Qwen3-0.6B through a trained
connector (plus LoRA in later stages) and tests whether answers depend on the image (correct, swapped, mismatched and
no-image conditions). Task 3 studies repeated sampling, answer selection, adaptive stopping and parallel vs. sequential
test-time compute on the supplied MATH and MathVista rollouts, plus a small live run with Qwen2.5-3B-Instruct.
All seeds are 6304.

| Task | Package | Status |
|---|---|---|
| 1. LLM fine-tuning (Full FT, L2-SP, LoRA-4/16, OPLoRA-16) | `task1_finetune/` | complete |
| 2. Vision-language model and visual grounding | `task2_vlm/` | complete (optional Step 7, the 49-token compression run, not done) |
| 3. Test-time compute: sampling, selection, adaptive inference | `task3_ttc/` | complete |

Other folders: `common/` shared helpers (seeding, atomic result files, run provenance, hashing, bootstrap), `configs/`
per-task YAML configurations, `splits/` fixed split and example IDs, `results/` result JSON/CSV/NPZ files, `figures/`
figures, `tests/` the test suite.

## Setup

### Mac (data preparation, offline analysis, figures, tests)

Python 3.11 (3.11.18 used), in a virtual environment at `.venv`:

```bash
python3.11 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements-lock.txt
```

```bash
export HF_HUB_DISABLE_XET=1
```

`requirements-lock.txt` is a `pip freeze` of the Mac environment (torch 2.15.0, transformers 5.21.0, peft 0.21.2,
accelerate 1.15.0, datasets 5.1.0, tokenizers 0.23.3, lm_eval 0.4.13, safetensors 0.8.0, math-verify 0.9.0).
`requirements.txt` holds only the frozen Task 3 scorer pins. `HF_HUB_DISABLE_XET=1` makes Hugging Face downloads use
plain HTTP. All modules run from the repository root as `python -m <package>.<module>`. Tests:

```bash
.venv/bin/python -m pytest -q
```

No training, full evaluation or model above 0.6B parameters runs on the Mac.

### Kaggle (training, evaluation, live generation)

Accelerator GPU T4 x2, Internet on, FP16 only (the T4 has no bf16 support). Each notebook clones this repository into
`/kaggle/working/atml-pa3`, sets `PYTHONUNBUFFERED=1` and `CUDA_VISIBLE_DEVICES`, keeps the preinstalled torch fixed with
a pip constraint, installs the pins below, and checks assets by sha256. The notebooks themselves are not committed; the
commands they ran are listed in the task sections below.

| Task | torch (preinstalled) | pip pins | Recorded in |
|---|---|---|---|
| 1 | 2.11.0+cu128 (CUDA 12.8) | transformers 5.21.0, peft 0.21.2, accelerate 1.15.0, datasets 5.1.0, safetensors 0.8.0, tokenizers 0.23.3, `lm_eval[hf]` 0.4.13 (evaluation) | torch/CUDA: `provenance.hardware` of `results/task1/train_*.json`, `results/task1/eval/*/scores.json`; lm_eval: `config.lm_eval.version` of the same scores files |
| 2 | 2.11.0+cu130 (CUDA 13.0) | transformers 5.21.0, peft 0.21.2, accelerate 1.15.0 (safetensors 0.8.0 as installed); then `pip uninstall -y torchao` | `versions` of `results/task2/main/final/eval.json` (`torchao: null`) and `results/task2/cache_features.json` |
| 3 (live) | 2.11.0+cu130 (CUDA 13.0) | transformers 5.21.0, accelerate 1.15.0, tokenizers 0.23.3, safetensors 0.8.0 | `results/task3/live/main/run.json` |

Task 1 run JSONs record torch, CUDA and lm_eval only; the other Task 1 pins are those of the setup cell. In Task 2 the
Kaggle image's torchao 0.10.0 makes peft 0.21.2 raise an ImportError in `get_peft_model`; torchao is not used, so it is
uninstalled before any run.

## Data

Raw datasets, image features and model checkpoints are not committed (`data/`, `cached/`, `checkpoints/` and
`outputs/` are gitignored). Every run JSON records the revisions and hashes it used.

### Task 1
- Model `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- Training data `meta-math/MetaMathQA` at revision `aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18`; the retained,
  GSM8K-decontaminated example IDs are in `splits/task1/train_ids.json` (sha256 `76ada292a58aff30f7a117597e82f656f4806c25877b8bcb8e4a46b6eedd1a96`).
- Evaluation through lm-evaluation-harness 0.4.13 task definitions: `gsm8k_cot` (openai/gsm8k main test,
  rev `740312add88f781978c0658806c59bc2815b9866`), `hellaswag` (Rowan/hellaswag validation, rev
  `218ec52e09a7e7462a5400043bb9a69a41d06b76`), `boolq` (aps/super_glue boolq validation, rev
  `3de24cf8022e94f4ee4b9d55a6f539891524d646`), `arc_easy` (allenai/ai2_arc ARC-Easy test, rev
  `210d026faf9955653af8916fad021475a3f00453`); revisions as recorded in `results/task1/eval/*/scores.json`.
- Probe set IDs (32 GSM8K test + 32 HellaSwag validation): `splits/task1/probe_ids.json`.
- Qualitative prompts: `first_plot_questions.yaml` from the emergent-misalignment repository, saved to
  `data/task1/first_plot_questions.yaml`, sha256 `215abde5b02811e922b6ac563ff33c490e0dff9ab58439df840f3135e9387173`:

```bash
curl -sSfL -o data/task1/first_plot_questions.yaml https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/evaluation/first_plot_questions.yaml
```

### Task 2
- Vision encoder `google/siglip-base-patch16-224` at revision `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`; language model
  `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`.
- COCO 2014 annotations and VQAv2 (URLs from https://visualqa.org/download.html), unzipped into
  `data/task2/raw/coco/annotations/` and `data/task2/raw/vqa/`:

| File | URL | sha256 |
|---|---|---|
| annotations_trainval2014.zip | http://images.cocodataset.org/annotations/annotations_trainval2014.zip | `031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009` |
| v2_Questions_Train_mscoco.zip | https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip | `05a64b6e2582d06d7585f5429674a9a33851878be1bff9f8668cdcf792df611e` |
| v2_Questions_Val_mscoco.zip | https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip | `e71f6c5c3e97a51d050f28243e262b28cd0c48d11a6b4632d769d30d3f93222a` |
| v2_Annotations_Train_mscoco.zip | https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip | `fb101bcefe91422c543c2bb6d70af11eb3119d0ff745ae283d09acdf66250853` |
| v2_Annotations_Val_mscoco.zip | https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip | `0caae7c8d1dafd852727f5ac046bc1efca9b72026bd6ffa34fc489f3a7b3291e` |
| v2_Complementary_Pairs_Train_mscoco.zip | https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip | `95b8142bd2e52bccfd0d9b15c4a9c580ee03745d04ef8481e54bc4ffb8b5c40d` |
| v2_Complementary_Pairs_Val_mscoco.zip | https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip | `4788add4425f17d301fbdbeb42b9ad855122ffc420ad23b5184af21dc3d2e716` |

- Images are fetched one by one from `http://images.cocodataset.org/{train2014,val2014}/<file_name>` (only the selected
  images; done by `task2_vlm.cache_features` on Kaggle and by `task2_vlm.analysis` for the qualitative figure).
- All selected image, question, caption and pair IDs, the full candidate orders and the mismatched-image mapping are in
  `splits/task2/` (sha256 of each file in `splits/task2/selection_report.json`).

### Task 3
- Supplied rollouts: the `ATML-PA3-Data` folder from the Google Drive link in the manual
  (https://drive.google.com/drive/folders/1utX-Gfi9vvaoi1mi0pzNpDGL1o1fAh23?usp=sharing), extracted to
  `data/task3/ATML-PA3-Data/`:

| File | Bytes | sha256 |
|---|---:|---|
| README.MD | 4,413 | `737f834ff66a031e226149066d97b7c38919f5c2f196906a5de037a42ac87241` |
| qwen25-math/generations.parquet | 26,932,880 | `ffdecd753a2c0183b811d5679f3cf15e91bce46d1aacb196e28afb553d7e1c84` |
| qwen25-math/probability_metrics.parquet | 155,221,097 | `48a492cd2c93bebfbd12b0b9df6a5783b611103e7a3cfefe010573ff6e200379` |
| qwen25vl-mathvista/generations.parquet | 17,098,339 | `dac12a7dc14baaa44e6c3ed8f8dfcf9d39b8493462c8a60ea162d0a66dda5435` |
| qwen25vl-mathvista/probability_metrics.parquet | 81,118,521 | `c1c7a9b9724aa4930673c1bb5759d2d82bacc57f4a530a83f533e9d2d43102ac` |
| qwen25vl-mathvista/attention_metrics.parquet | 138,497,144 | `4f6cfce99519808c75f950e53c957795697d8b503a7fd407fcfdd260f06d0058` |

- MathVista metadata `data/task3/mathvista/testmini.json` from lupantech/MathVista at commit
  `5f6b4deca69ecd1b8a1a4f338cee085e50524e4b` (sha256 `ab0148dca9d401c31cc47f29b3826eec28a76ff155a1c0878d81e8daf0413480`,
  checked against `configs/task3/offline.yaml`).
- Live model `Qwen/Qwen2.5-3B-Instruct` at revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- Dev/eval splits: `splits/task3/{math,mathvista}_split.json`; the 50 live problems: `splits/task3/live_problems.json`
  (sha256 `deac8bc4c27cb5509526e8300820a7836d02f35b9f7108108a8f6cad28885cd5`).

## Task 1: LLM Fine-Tuning

Configurations: `configs/task1/{full_ft,l2sp,lora4,lora16,oplora16,eval}.yaml`. Commands in run order.

Mac, fixed IDs (both refuse to change an existing file):

```bash
.venv/bin/python -m task1_finetune.make_probe_set
```

```bash
.venv/bin/python -m task1_finetune.make_train_ids
```

Kaggle, FP16 check and smoke runs (3 updates per method; GPU chosen with `CUDA_VISIBLE_DEVICES`):

```bash
python -m task1_finetune.fp16_check
```

```bash
python -m task1_finetune.train --config configs/task1/full_ft.yaml --smoke_updates 3
```

(same for `l2sp`, `lora4`, `lora16`, `oplora16`), then:

```bash
python -m task1_finetune.status --table results/task1/smoke_full_ft.json results/task1/smoke_l2sp.json results/task1/smoke_lora4.json results/task1/smoke_lora16.json results/task1/smoke_oplora16.json
```

Kaggle, training (two GPU queues started 60 s apart; each run is
`python -m task1_finetune.train --config configs/task1/<method>.yaml --recovery_dir /kaggle/tmp/task1_recovery/<method>`,
followed by `python -m task1_finetune.status --verify results/task1/train_<method>.json`):

```bash
bash task1_finetune/run_queue.sh 0 full_ft lora4 oplora16
```

```bash
bash task1_finetune/run_queue.sh 1 l2sp lora16
```

```bash
python -m task1_finetune.status --train_table results/task1/train_full_ft.json results/task1/train_l2sp.json results/task1/train_lora4.json results/task1/train_lora16.json results/task1/train_oplora16.json
```

An interrupted run resumes with
`python -m task1_finetune.train --config configs/task1/<method>.yaml --recovery_dir /kaggle/tmp/task1_recovery/<method> --resume`.

Kaggle, evaluation (a second notebook with the training output attached as input; the checkpoints are found and
checked against the sha256 values in the train JSONs):

```bash
python -m task1_finetune.status --find_ckpts /kaggle/input
```

```bash
python -m task1_finetune.evaluate --condition baseline --smoke --limit 16 --qual_limit 2
```

```bash
bash task1_finetune/run_eval_queue.sh 0 baseline full_ft_u0625 full_ft_u1250 l2sp_u1250
```

```bash
bash task1_finetune/run_eval_queue.sh 1 lora4_u1250 lora16_u1250 oplora16_u1250 diagnostics
```

Each queue entry is `python -m task1_finetune.evaluate --condition <condition>` or `python -m task1_finetune.diagnostics`.
Then:

```bash
python -m task1_finetune.status --eval_summary
```

```bash
python -m task1_finetune.status --collect_qualitative
```

Mac, analysis and figures (reads committed results and the hand-filled `results/task1/annotations.csv`):

```bash
.venv/bin/python -m task1_finetune.analyze
```

Results: `results/task1/train_<method>.json` (config, per-update log, resources), `smoke_*.json`, `fp16_check.json`,
`eval/<condition>/{scores.json,gsm8k,hellaswag,boolq,arc_easy}.jsonl.gz` (per-example outputs),
`diagnostics.json`, `oplora_verification.json`, `qualitative/` (168 responses), `annotations.csv`, and
`analysis/*.json` (benchmarks with paired CIs, resources, learning, diagnostics tables). `task1_finetune.analyze` writes
`results/task1/analysis/` and rebuilds `figures/task1/{learning_curves,benchmark_changes,diagnostics}.png`.

## Task 2: Vision-Language Models and Visual Grounding

Configurations: `configs/task2/{data,categories,features,train,qualitative_rule}.yaml`. Commands in run order.

Mac, data selection and label check (needs the raw COCO/VQAv2 JSONs above):

```bash
.venv/bin/python -m task2_vlm.select_data
```

```bash
.venv/bin/python -m task2_vlm.label_sheets
```

```bash
.venv/bin/python -m task2_vlm.apply_label_flags
```

`label_sheets` draws the grounding evaluation items for a manual label check; `apply_label_flags` replaces flagged items
with the next candidates in the saved order and rewrites the affected split files.

Kaggle (GPU 0 only): LoRA preflight, feature caching, smoke, main run:

```bash
python -m task2_vlm.preflight
```

```bash
python -m task2_vlm.cache_features --device cuda --images-dir /kaggle/tmp/task2_images --out-dir /kaggle/working/features --result results/task2/cache_features.json
```

```bash
python -m task2_vlm.train --smoke --features_dir /kaggle/working/features --recovery_dir /kaggle/tmp/task2_recovery_smoke
```

```bash
python -m task2_vlm.evaluate --smoke --features_dir /kaggle/working/features
```

```bash
bash task2_vlm/run_main.sh
```

`run_main.sh` runs `python -m task2_vlm.train --features_dir /kaggle/working/features --recovery_dir /kaggle/tmp/task2_recovery`
(three stages, 4,500 updates, dev evaluation during training) and then the final evaluation of the Stage 3 endpoint,
`python -m task2_vlm.evaluate --features_dir /kaggle/working/features`. An interrupted run resumes with
`python -m task2_vlm.train --resume --features_dir /kaggle/working/features --recovery_dir /kaggle/tmp/task2_recovery`.
Progress and summary:

```bash
python -m task2_vlm.status --summary --run_dir results/task2/main
```

Mac, analysis and figures:

```bash
.venv/bin/python -m task2_vlm.analysis
```

Results: `results/task2/cache_features.json`, `results/task2/main/train.json` (config, calibration, per-update log,
dev summaries), `results/task2/main/dev/`, `results/task2/main/final/{eval.json,examples.json}` (test suites under
correct, swap, mismatched and no-image conditions, paired bootstrap CIs), `results/task2/analysis/*.json`.
`task2_vlm.analysis` writes `results/task2/analysis/` and rebuilds `figures/task2/learning_curves.{pdf,png}` and
`figures/task2/qualitative_main.{pdf,png}`; it downloads the few val images the qualitative figure shows unless
`--no_download` is given.

## Task 3: Test-Time Compute

Configuration: `configs/task3/offline.yaml`. Commands in run order.

Mac, offline (supplied rollouts):

```bash
.venv/bin/python -m task3_ttc.splits --dataset math
```

```bash
.venv/bin/python -m task3_ttc.prepare --dataset math
```

```bash
.venv/bin/python -m task3_ttc.offline --dataset math
```

```bash
.venv/bin/python -m task3_ttc.adaptive select --dataset math
```

```bash
.venv/bin/python -m task3_ttc.adaptive evaluate --dataset math
```

Run each of these with `--dataset mathvista` as well. `splits` checks/writes `splits/task3/<dataset>_split.json`;
`prepare` forms answer groups and correctness labels (`results/task3/rollouts/`, `results/task3/prepare_<dataset>.json`);
`offline` runs Steps 1-3 (`results/task3/offline/`, `results/task3/signals/`); `adaptive select` chooses thresholds on
the dev split only (`results/task3/adaptive/<dataset>_{dev_grid,selected}.json`), and `adaptive evaluate` applies the
frozen settings to the eval split (`<dataset>_eval.json`, `<dataset>_eval_trace.npz`). Then:

```bash
.venv/bin/python -m task3_ttc.figures data
```

```bash
.venv/bin/python -m task3_ttc.figures plot
```

```bash
.venv/bin/python -m task3_ttc.qualitative
```

Mac, live problem file (50 MATH problems, problem text only):

```bash
.venv/bin/python -m task3_ttc.live_problems
```

Kaggle (GPU 0 only), live generation (no gold answer is read on Kaggle):

```bash
python -m task3_ttc.live --preflight --out results/task3/live/preflight
```

```bash
python -m task3_ttc.live --out results/task3/live/smoke --questions 1
```

```bash
python -m task3_ttc.live --out results/task3/live/main
```

```bash
python -m task3_ttc.live_status --summary --run results/task3/live/main
```

Mac, live analysis and examples:

```bash
.venv/bin/python -m task3_ttc.live_analysis
```

```bash
.venv/bin/python -m task3_ttc.live_examples
```

Results: `results/task3/rollouts/`, `results/task3/offline/` (summaries, per-question CSVs, selections NPZ, paired
bootstraps), `results/task3/signals/`, `results/task3/adaptive/`, `results/task3/figdata/mathvista_figdata.json`,
`results/task3/qualitative/{examples,live_examples}.json`, `results/task3/live/` (`main/{calls.jsonl,run.json}`,
`preflight/`, `download.json`, `analysis.json`, `per_question.csv`). Figures: `figures/task3/F1-F8*.pdf`.
`task3_ttc.figures plot` rebuilds them from committed result files only (it checks that its inputs are tracked and
unmodified); `task3_ttc.figures data` rebuilds `results/task3/figdata/mathvista_figdata.json` (F5, F6) and needs the
supplied parquet files and the token cache written by `prepare`.

## Notes

- `scorer_version`: the offline results (`prepare_*.json`, `offline/*_summary.json`, `adaptive/*_eval.json`) record
  `1fac30ddd0981fb05724e7892f27d9b4fbff3da31db7e2a7b33da86ccc3dcea8`; `results/task3/live/analysis.json` records
  `a6b3433a1382548f67b84430ae96f07df8e344bd92f1c3ea481b7ac764f58738`. The scorer code and libraries are the same
  (same `answers.py` sha256); the live id differs only because the live analysis is MATH-only and its version record
  has no MathVista `testmini_sha256`.
- `results/task3/offline/*_selections.npz` holds an object array (`split`), so load it with
  `np.load(path, allow_pickle=True)`.
- `TASK3_LIVE_TEST_KILL_AFTER_CALLS` is used by the tests only (the process kills itself after that many new calls to
  test resuming). Unset, it is 0 and nothing is killed; it was not set in any real run.
- Run JSONs record the git commit and dirty flag, hardware, dtypes, wall-clock time and peak memory of the run that
  wrote them; a result file is never overwritten by a different run.

## Attribution

- lm-evaluation-harness (EleutherAI), version 0.4.13: Task 1 uses its task definitions, prompts, answer filters and
  metrics as a library (`gsm8k_cot`, `hellaswag`, `boolq`, `arc_easy`); `task1_finetune/tracking_lm.py` subclasses its
  `HFLM` model wrapper.
- emergent-misalignment: the Task 1 qualitative prompts are `evaluation/first_plot_questions.yaml` from
  https://github.com/emergent-misalignment/emergent-misalignment (downloaded, not committed).
- VQA answer normalization in `task2_vlm/vqa_norm.py` is a port of the official evaluator
  `PythonEvaluationTools/vqaEvaluation/vqaEval.py` from https://github.com/GT-Vision-Lab/VQA at commit
  `a013f0043c1e2cdc995922dfe257f7149aa9af06`.
- MathVista answer normalization in `task3_ttc/answers.py` is a port of `normalize_extracted_answer` and the scoring
  rules in `evaluation/calculate_score.py` from https://github.com/lupantech/MathVista at commit
  `5f6b4deca69ecd1b8a1a4f338cee085e50524e4b`.
- MATH answer equivalence uses math-verify 0.9.0 (with latex2sympy2_extended 1.11.0, sympy 1.14.0).
- Models are loaded and adapted with Hugging Face transformers and peft (LoRA); datasets and model snapshots come from
  the Hugging Face Hub.
