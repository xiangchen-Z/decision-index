# Ainvest-Decision-27b-v1.0

**Only text decision capabilities were trained. This is a text-board submission; no vision fine-tuning or vision evaluation is claimed.**

Please evaluate **Ainvest-Decision-27b-v1.0** on the maintainers' private same-skills and new-domain tests for the Decision Index. This PR adds one model registration to `submissions/README.md`; it does not change the benchmark or scoring code.

- Model / BF16 weights / inference package: https://huggingface.co/Ainvest-World/Ainvest-Decision-27b-v1.0/tree/96599823ab01c9c06b61598ee38704d454bc3b71
- Complete public scores (`complete: true`): https://huggingface.co/datasets/Ainvest-World/Ainvest-Decision-27b-v1.0-decision-index-results/blob/c4199e6304a5cd19d6eb2bcb44d1870fc15d3226/runs/m02/scores.json
- Raw result file: https://huggingface.co/datasets/Ainvest-World/Ainvest-Decision-27b-v1.0-decision-index-results/resolve/c4199e6304a5cd19d6eb2bcb44d1870fc15d3226/runs/m02/results.jsonl.gz
- Model/result identity and file hashes: https://huggingface.co/datasets/Ainvest-World/Ainvest-Decision-27b-v1.0-decision-index-results/blob/c4199e6304a5cd19d6eb2bcb44d1870fc15d3226/publication.json
- Adapter code revision: the model Hub commit `96599823ab01c9c06b61598ee38704d454bc3b71`, package `ainvest_decision.engine:AinvestDecisionEngine`.

### Running the submitted model

In a Python 3.12/CUDA environment, install a CUDA-appropriate PyTorch 2.14.0 build and the pinned dependencies. `causal-conv1d` may need a matching CUDA build environment. The model is about52.2 GB in BF16.

```bash
python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Ainvest-World/Ainvest-Decision-27b-v1.0", revision="96599823ab01c9c06b61598ee38704d454bc3b71"))'
# Set this to the snapshot directory printed by snapshot_download.
AINVEST_MODEL_DIR=/absolute/path/to/snapshot
pip install -r "$AINVEST_MODEL_DIR/requirements.lock.txt"
pip install --no-deps "$AINVEST_MODEL_DIR"
```

With the Decision Index reproduction kit and this model package installed:

```bash
python -m decision_index run --edition 0.3   --engine ainvest_decision.engine:AinvestDecisionEngine   --option model=Ainvest-World/Ainvest-Decision-27b-v1.0   --option revision=96599823ab01c9c06b61598ee38704d454bc3b71   --option device=cuda:0 --option dtype=bfloat16   --option max_length=16384 --option batch_size=16   --option batch_tokens=16384 --option sort_window=256   --rows /path/to/locally-rebuilt-public-rows.jsonl.gz --out runs/ainvest
```

The Python interface accepts exactly `state` and typed `questions` and returns `(response, raw)`. Every supplied option receives a probability; `noul` returns P(true). It uses the fixed evaluated rendering, noncausal full-attention, last pooling, the trained 255-way readout, and the original worker's post-load BF16 floating-buffer cast. It is not an AutoModelForCausalLM generation checkpoint.

### Original complete public run

- Public 0.3 suite: **140,178 requests / 312,320 question outcomes**, Decision Index **71.88**, raw index **78.57**.
- 140,176 requests answered; **2 unsupported requests / 33 unsupported question outcomes**, retained unchanged.
- Original full run: 8 H200 replicas; PyTorch 2.14.0+cu126, Transformers 5.17.0, BF16; bucketed backend; batch cap 16, padded token cap 16,384, sort window 256, warmup 1, OMP threads 4. No truncation, compilation, CUDA graphs or calibration.
- Original scorer engine label `packing`, result rows `decision-bucketed`, response.model checkpoint identity `0a6ee4165dd6d686c0533cfccb3b1d3f084993b0d49708f1a84ea361aa71c9fd`; the publication manifest maps these to the submitted Hub revision.
- Original test_classifier.py source SHA256 `a55c1b6c0d96395738d0f0c90524da4860b0b65169dbb6fb52083374827bf27c`; original worker.py source SHA256 `5fa893713321c05780d8b918cc3d05112647b76542e97fd1c40815e94ae2aca7`.
- Native score and independent offline rescore already matched. Original result and scoring bytes are preserved; no full public rerun or new full rescore was done for publication.

### Publication validation and latency

A fixed input-only selected 100-request sample, all 611 subquestions, was checked on a single H200. No selected answer flips occurred; input hashes, lengths, option identities and typed responses matched. Probability drift against the original full-run batching stayed within an original-worker replay baseline frozen before candidate evaluation. The sample and reports are included under `validation/` in the Dataset.

Local single-H200 synchronized request wall time, including rendering/tokenization/transfers and every question in each request: median **141.67ms**, mean **994.69ms**, p80 **1047.76ms**; peak allocated **55.69GB**. These are local sample measurements, not the maintainers' RTX PRO 6000 measurement and not a claim of passing the official latency rule. The historical full-run timing uses amortized batch-forward time and is labeled separately.

### Capacity and licensing

Declared capacity is 2–255 options and 16,384 tokens for each complete rendered question input. If any question exceeds capacity, the entire request is unsupported; no option is dropped and nothing is truncated. The published release is text-only.

Weights and Ainvest-authored code are Apache-2.0 with upstream notices retained. Evaluation inputs and gold labels are not redistributed. We are requesting that you run your private evaluations; we are not requesting access to the private sets.
