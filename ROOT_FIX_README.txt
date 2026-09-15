ECAPA-TDNN ex2 frozen handoff root fix v2
=========================================

This patch supersedes ex2_frozen_handoff_patch.zip.

Root fix included:
- The primary ExperimentProvenance/validation_trials.parquet stores the binary
  verification label in column `label`, not `target`.
- src/frozen_handoff_cache.py now detects `label` as the canonical frozen
  handoff column, accepts legacy `target` for backward compatibility, then
  normalizes internally to `target` so scoring/training code does not change.
- This is a repository-source fix. Notebook 1 and Notebook 2 should NOT patch
  source files at runtime.

Apply from repository root:
  unzip -o ex2_frozen_handoff_patch_v2.zip -d <repo-root>
  git add requirements.txt requirements-cuda.txt \
      scripts/build_adaptive_augmented_3s_fbank_cache.py \
      src/adaptive_augmented_3s_training.py \
      src/frozen_handoff_cache.py \
      tests/test_frozen_handoff_cache.py
  git commit -m "Fix frozen validation parquet label schema"
  git push origin ex2

After pushing:
- start a fresh Colab runtime or force-reclone branch ex2;
- existing shared validation FBank cache on Google Drive is reused;
- no validation FBank rebuild is required;
- Notebook 2 will clone the same corrected branch and therefore uses the same
  root fix automatically.

Relevant unit test result in the patch workspace:
  5 passed
