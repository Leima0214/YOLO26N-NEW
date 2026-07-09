# Security and Reproducibility

## Security rules

### Never commit

- `*.pt`, `*.pth`, `*.onnx`, `*.engine` (model weights)
- `datasets/`, `datasets_derived/`, `runs/` (data and results)
- `*.jpg`, `*.png`, `*.bmp` except those in `ultralytics/assets/`
- `*.tar.gz`, `*.zip` archives
- API keys, tokens, passwords, SSH keys
- `wandb/` API keys and logs
- `.env` files

### Code safety

- ✅ Use `yaml.safe_load()` (never `yaml.load()`)
- ✅ Use `pathlib.Path` (never raw string path manipulation)
- ✅ Use `subprocess.run()` with `shell=False` and list arguments
- ❌ Never use `eval()` or `exec()`
- ❌ Never use `os.system("rm -rf " + var)`
- ❌ Never auto-delete `runs/` or `datasets/`
- ❌ Never load `.pt` files from untrusted sources
- ❌ Never commit GitHub tokens or platform credentials

### Training safety

- Always run `check_dataset.py` before training
- Always run 3-epoch smoke test before full run
- Use `tmux` or `nohup` for long training jobs
- Log stdout/stderr: `| tee logs/{name}.log`
- Use unique `--name` with date: `_rYYYYMMDD`

## Reproducibility checklist

### Per experiment

- [ ] Commit hash recorded (`git rev-parse HEAD`)
- [ ] Data config YAML committed
- [ ] Training command documented in `commands.md`
- [ ] `results.csv` saved
- [ ] `args.yaml` saved
- [ ] Environment info recorded (`torch.__version__`, CUDA version, GPU model, `nvidia-smi`)
- [ ] No uncommitted local changes during training

### Per paper

- [ ] All configs in `configs/`
- [ ] All scripts in `scripts/`
- [ ] All results in `experiments/{name}/`
- [ ] Training equivalence audit in `experiments/direct_train_equivalence.md`
- [ ] Dataset protocol in `docs/paper1_japan7_yolo26n/dataset_protocol.md`
- [ ] Experiment protocol in `docs/paper1_japan7_yolo26n/experiment_protocol.md`

## Git workflow

```
GPU Server:  training + save results → push light artifacts
Codex:       docs, code, configs → push from local
Local:       pull both, check consistency, backup tags
```

Before any push:
```bash
git fetch
git pull --rebase
git status           # check for conflicts
git diff --stat      # check for large files
```

Never force push unless confirmed no data loss.

## Disaster recovery

- `japan7_best_weights_20260707.tar.gz` on remote server (best.pt backups)
- GitHub has all code, configs, and light results
- Remote server has full `runs/` directory
- Local has working copy

If GitHub is lost: re-clone from remote server's local repo.
If remote runs are lost: re-train from committed scripts + configs.
