# Vietnamese Speaker Verification - ECAPA-TDNN + AAM-Softmax

This repository fine-tunes the pretrained
`speechbrain/spkrec-ecapa-voxceleb` encoder with cached SpeechBrain FBank
features, P x K speaker-balanced sampling, and a project-owned AAM-Softmax
head. Model selection uses EER on fixed validation trials.

## Frozen common manifest

The supported input is the provenance-rich `manifest.csv` produced by the data
pipeline. The training path reads these fields directly:

```text
canonical_audio_id
canonical_wav_relpath
canonical_speaker_id
split
source_dataset
global_speaker_id
source_recording_id
sample_id
start_sec
end_sec
speech_ratio
sample_rate
num_samples
duration_sec
padded
```

Additional columns are preserved by the authoritative source file and ignored
by training. Existing `split=train|validation` assignments are frozen and are
never regenerated. `global_speaker_id` is the training identity. Train labels
are generated deterministically as contiguous values from zero; validation
labels are `-1`.

Windows path separators in `canonical_wav_relpath` are converted in memory to
portable `/` separators. The dataset root must contain the `train/` and
`validation/` directories referenced by those paths.

## Dynamic values

Dataset-dependent values are derived from the manifest:

- train and validation row counts;
- train and validation speaker counts;
- AAM number of classes;
- contiguous speaker labels;
- batches per epoch;
- cosine-scheduler steps;
- validation trial count;
- cache and checkpoint identity bindings.

The model/training contract remains fixed: ECAPA embedding size 192,
AAM margin 0.2 and scale 30, P=16, K=2, logical batch 32, microbatch 4,
AdamW, ECAPA/AAM learning rates `1e-5`/`1e-3`, weight decay `1e-4`, AMP,
cosine decay, and validation-EER early stopping.

## Google Colab paths

Keep FBank shards on fast Colab local storage and checkpoints on Drive:

```text
/content/data/common_raw/                  dataset root
/content/data/manifest.csv                 frozen manifest
/content/fbank_cache/                      local FBank cache
/content/drive/MyDrive/ecapa_results/run1  persistent checkpoints
```

The code has no fixed Windows, Kaggle, Colab, or Drive paths. Supply every
machine-specific location through the CLI.

## 1. Generate frozen validation trials

```bash
python scripts/generate_adaptive_augmented_3s_validation_trials.py \
  --manifest /content/data/manifest.csv \
  --output /content/validation_trials.csv \
  --identity-output /content/validation_trials_identity.json \
  --seed 2026 \
  --genuine-trials 10000 \
  --impostor-trials 10000 \
  --overwrite
```

Genuine pairs prefer two different `(source_dataset, source_recording_id)`
values. Every experiment must reuse the same frozen trial file.

## 2. Build raw SpeechBrain FBank shards

```bash
python scripts/build_adaptive_augmented_3s_fbank_cache.py \
  --manifest /content/data/manifest.csv \
  --dataset-root /content/data/common_raw \
  --cache-root /content/fbank_cache \
  --device cuda:0 \
  --batch-size 64 \
  --shard-size 256
```

The cache contains pre-normalization SpeechBrain FBank tensors `[301, 80]`.
It is bound to the SHA-256 of the supplied manifest and the derived speaker
label mapping. Use a new empty cache directory after changing the manifest.

## 3. Disposable dry run

```bash
python scripts/run_adaptive_augmented_3s_training.py \
  --dry-run \
  --manifest /content/data/manifest.csv \
  --validation-trials /content/validation_trials.csv \
  --cache-root /content/fbank_cache \
  --output-dir /content/ecapa_dry_run \
  --device cuda:0
```

Use an empty dry-run output directory. The command performs one fresh update
and one resumed update.

## 4. Full training

```bash
python scripts/run_adaptive_augmented_3s_training.py \
  --run \
  --manifest /content/data/manifest.csv \
  --validation-trials /content/validation_trials.csv \
  --cache-root /content/fbank_cache \
  --output-dir /content/drive/MyDrive/ecapa_results/run1 \
  --device cuda:0
```

`best.pt` is selected by validation EER. `last.pt` stores the resumable state.

Resume the same run with:

```bash
python scripts/run_adaptive_augmented_3s_training.py \
  --run \
  --manifest /content/data/manifest.csv \
  --validation-trials /content/validation_trials.csv \
  --cache-root /content/fbank_cache \
  --output-dir /content/drive/MyDrive/ecapa_results/run1 \
  --resume /content/drive/MyDrive/ecapa_results/run1/last.pt \
  --device cuda:0
```

A checkpoint refuses resume when the manifest, trials, cache, or training
configuration differs from the original run.

## Optional derived artifacts

The following command creates compact train/validation manifests, a label map,
split identities, and trials for audit or inspection. It reuses the existing
split and never repartitions speakers:

```bash
python scripts/create_adaptive_augmented_3s_package.py \
  --manifest /content/data/manifest.csv \
  --dataset-root /content/data/common_raw \
  --check-audio-exists \
  --overwrite
```

## Final-test isolation

Final test remains a separate dataset with identities disjoint from train and
validation. Do not use it to select augmentation, hyperparameters, checkpoint,
trials, or threshold. Only the validation-selected `best.pt` may be used for
final evaluation.
