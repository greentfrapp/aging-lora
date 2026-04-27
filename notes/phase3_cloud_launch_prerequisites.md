# Phase-3 cloud launch prerequisites

Concrete prerequisites to run Phase-3 fine-tunes on AWS EC2 g5.xlarge (per `notes/phase3_cloud_compute_review.md`). Ordered by dependency; mark each item before launching the first cloud run.

## 1. Mid-run checkpointing in `train_loop.py` (~30 LOC code change)

**Why:** unblocks spot strategy for the Phase-3-B sweep (saves ~60% on cost). Without it, a spot interruption mid-run loses 4+ hours of work.

**Spec:**

- Add `TrainConfig.ckpt_every_steps: int = 0` (0 = disabled, current behaviour) and `TrainConfig.resume_from: Path | None = None`.
- After every `ckpt_every_steps` optimizer steps in `train()`, serialize trainable state (LoRA + head, same filter as the end-of-training checkpoint) to `ckpt_path.with_suffix('.partial.pt')` plus a JSON sidecar `ckpt_path.with_suffix('.partial.meta.json')` containing `{global_step, epoch, optimizer_state_dict_path, rng_state}`. Atomic rename pattern (write to `.tmp`, rename) to survive interruption mid-write.
- On launch, if `resume_from` is set: load LoRA + head weights, load optimizer + RNG state, fast-forward the dataloader to skip already-seen batches, and resume from `global_step + 1`.
- Add CLI flags `--ckpt-every-steps INT --resume-from PATH`.
- Recommended interval: `ckpt_every_steps=50` (one save every ~20 min on A10g) — bounds interruption loss to ≤20 min.

**Acceptance:** kill an in-progress training at step 100, restart with `--resume-from`, verify the JSONL log shows `step=101` resuming from the same train_mse trajectory and the final eval matches a non-resumed run within seed-noise.

## 2. S3 staging of code + data + FM checkpoints

**Why:** Repeated EC2 launches need a fast way to populate code + ~20 GB of data/checkpoints. Pulling from S3 to EBS at boot is ~5 min vs ~30 min to download from original sources.

**S3 prefix layout** (one-time setup, then reused by every launch):

```
s3://<your-bucket>/phase3/
├── code/                                    # synced from local repo on each launch
│   └── (mirrors repo contents excluding gitignored data/save)
├── data/cohorts/integrated/                 # ~5–15 GB of harmonized h5ads
│   ├── B.h5ad
│   ├── CD4p_T.h5ad
│   ├── CD8p_T.h5ad
│   ├── Monocyte.h5ad
│   └── NK.h5ad
├── data/cohorts/aida_eval/                  # AIDA eval split for Phase-3 AIDA pass
│   └── *.h5ad
├── save/Geneformer/                         # ~440 MB
├── save/scGPT_human/                        # ~200 MB
├── save/scFoundation/                       # ~6 GB compressed
└── save/UCE/                                # ~5.6 GB
```

**One-time bucket setup** (user-side):

```bash
# Create bucket (one-time):
aws s3 mb s3://<your-bucket> --region us-east-1

# Upload data + checkpoints (one-time, ~30 min):
aws s3 sync data/cohorts/integrated/ s3://<your-bucket>/phase3/data/cohorts/integrated/
aws s3 sync data/cohorts/aida_eval/ s3://<your-bucket>/phase3/data/cohorts/aida_eval/
aws s3 sync save/ s3://<your-bucket>/phase3/save/

# Per-launch code sync (cheap, ~30 sec):
aws s3 sync . s3://<your-bucket>/phase3/code/ \
    --exclude ".venv/*" --exclude ".git/*" --exclude "data/*" --exclude "save/*" \
    --exclude "logs/*" --exclude "results/*" --exclude "scratchpad/*"
```

**IAM:** create an EC2 instance role with `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` on `<your-bucket>/phase3/*`. Attach to launched instances.

## 3. AMI choice

**Recommended:** **Deep Learning AMI GPU PyTorch 2.x (Ubuntu 22.04)** — comes with CUDA 12.x, cuDNN, NVIDIA driver pre-installed. Avoids the 30-minute setup cost of stock Ubuntu.

Search: `aws ec2 describe-images --owners amazon --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.*Ubuntu 22.04*"` (sort by date, take latest).

Then `uv` is the only thing we need on top — `pip install uv` and `uv sync` on first launch.

## 4. Bootstrap script

User-data script that runs on instance start. Saves to `scripts/cloud_bootstrap.sh` (one-time prepared, then reused).

**Spec:**

```bash
#!/bin/bash
set -euo pipefail

REGION=us-east-1
BUCKET=<your-bucket>
WORKDIR=/home/ubuntu/phase3

# 1. Pull code + data + checkpoints from S3 in parallel
mkdir -p "$WORKDIR" && cd "$WORKDIR"
aws s3 sync s3://$BUCKET/phase3/code/ ./ --quiet &
aws s3 sync s3://$BUCKET/phase3/data/   ./data/   --quiet &
aws s3 sync s3://$BUCKET/phase3/save/   ./save/   --quiet &
wait

# 2. Set up Python environment via uv
pip install uv
uv venv --python 3.12
uv sync

# 3. Verify GPU is available
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

# 4. Hand off — actual run command passed via SSM or appended to user-data
# (see launch wrapper §5)
```

**Estimated boot-to-ready time:** ~5 min on g5.xlarge (S3-to-EBS bandwidth is ~250 MB/s within the same region).

## 5. Launch wrapper

User-side script that:
1. Syncs the latest local code to S3 (`aws s3 sync`).
2. Spawns a g5.xlarge with bootstrap user-data.
3. Tails the SSM session for the run's stdout/JSONL.
4. Pulls results back to local on completion (`aws s3 sync` from results-out prefix).
5. Terminates the instance.

**Concrete first version** at `scripts/cloud_run.sh`:

```bash
#!/bin/bash
# Usage: scripts/cloud_run.sh <run-name> -- <full uv run python -m src.finetune.cli ARGS>
set -euo pipefail

RUN=$1; shift
[[ "$1" == "--" ]] && shift  # drop separator

# 1. Stage code
aws s3 sync . s3://<your-bucket>/phase3/code/ --exclude ".venv/*" --exclude ".git/*" \
    --exclude "data/*" --exclude "save/*" --exclude "logs/*" --exclude "results/*" --exclude "scratchpad/*"

# 2. Launch instance with bootstrap + run command
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id ami-<dlami-id> \
    --instance-type g5.xlarge \
    --iam-instance-profile Name=phase3-s3-access \
    --instance-market-options 'MarketType=spot' \
    --user-data "$(cat scripts/cloud_bootstrap.sh; echo; echo cd /home/ubuntu/phase3; echo $@; \
                   echo aws s3 sync results/ s3://<your-bucket>/phase3/results/$RUN/; \
                   echo sudo shutdown -h +1)" \
    --query 'Instances[0].InstanceId' --output text)

echo "Launched $INSTANCE_ID for run=$RUN"
echo "Tail with: aws ssm start-session --target $INSTANCE_ID"
echo "Pull results with: aws s3 sync s3://<your-bucket>/phase3/results/$RUN/ results/cloud/$RUN/"
```

**Per-run cost:** g5.xlarge spot ~$0.35/h × 4 h E5b = ~$1.40, on-demand ~$4.

## 6. Cost monitoring + cleanup

- **Tag every instance** with `Project=phase3,Run=<run-name>` for cost-explorer attribution.
- **Auto-shutdown**: bootstrap script ends with `sudo shutdown -h +1` so an orphaned instance dies after the run completes (or 1 min after, to give result-sync time). Costs ~$0 to leave one instance permanently buggy if shutdown fails — set up CloudWatch alarm on EC2 hours per day in the project tag.
- **Spot interruption notifications**: subscribe an SNS topic to spot-interruption events; the receiving Lambda can auto-respawn from the latest mid-run checkpoint.

## 7. Phase-3-A vs Phase-3-B prerequisite gate

Phase-3-A (E5b, E5c, possibly more diagnostic single runs) can use this stack at minimal viable scope:

- **Required**: §2 S3 staging, §3 AMI choice, §4 bootstrap, §5 launch wrapper.
- **Not required for Phase-3-A**: §1 mid-run checkpointing (single on-demand runs are fine without it).

Phase-3-B (18-fine-tune sweep) needs the full stack:

- All of the above PLUS §1 mid-run checkpointing (so spot is viable).

## 8. Checklist

- [ ] §1 Implement mid-run checkpointing in `train_loop.py` (Phase-3-B prerequisite)
- [ ] §2 Create S3 bucket, upload data + FM checkpoints (one-time, ~30 min)
- [ ] §2 IAM instance role with S3 r/w on `phase3/*`
- [ ] §3 Pin a specific Deep Learning AMI ID
- [ ] §4 Write `scripts/cloud_bootstrap.sh`
- [ ] §5 Write `scripts/cloud_run.sh` launch wrapper
- [ ] §6 Cost-monitoring tag conventions + CloudWatch alarm
- [ ] First trial run: launch E5b on g5.xlarge on-demand, verify the full pipeline end-to-end before committing to spot or to the Phase-3-B sweep
