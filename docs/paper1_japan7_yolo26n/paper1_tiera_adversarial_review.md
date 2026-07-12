# Paper 1 Tier A Adversarial Review

Date: 2026-07-11

Scope: the A01-A12 YAML generator, custom modules, YOLO parser integration, pretrained transfer, and pilot launcher. This review used the previous AI analysis as a lead list, then independently traced and tested the implementation. It did not train any model.

## Findings And Corrections

### Critical

1. **Attention modules changed pretrained features at initialization.** EMA, SE, CBAM, and FFA previously applied a non-identity gate immediately after loading baseline weights. They now use a zero-initialized residual scale, so the initial function is exactly the inherited YOLO26n function and the new path is learned gradually.
2. **Index-based weight remapping overstated transfer quality.** Counting state-dict items did not account for tensor size, inserted-layer offsets, or the extra P2 Detect branch. Every generated YAML now carries a semantic source-to-target map. The launcher reports parameter coverage for the whole model and separately for backbone, neck, and Detect. P2 models map baseline Detect branches P3/P4/P5 to target branches 1/2/3 and leave only the new P2 branch uninitialized.
3. **FDConv violated the replaced Conv contract.** The old path could lose BatchNorm/activation behavior, ignored important stride assumptions, cached a fixed-size frequency mask, and had fragile mixed-precision behavior. The corrected module preserves Conv-BN-SiLU state names, creates a size-correct FFT mask per input, computes FFT work in float32, bounds the residual gain, and starts as an exact standard Conv.
4. **BiFPN used a large unrelated dependency surface and ambiguous parser semantics.** `tasks.py` imported the entire `v9.py`, which transitively loaded packages such as pandas and IPython. The replacement is a minimal positive weighted-concat module; the parser injects the input count explicitly, validates shape/count, and initializes to ordinary concatenation. It is BiFPN-style weighting, not classic additive BiFPN, and must be described that way in the paper.
5. **Run/report handling was unsafe under concurrency or hostile paths.** Run names could collide, reports could be written concurrently, and model/data paths were insufficiently constrained. The launcher now validates bounds and names, confines input files to the repository, reserves run directories atomically, locks report updates with stale-lock recovery, and rejects duplicate or unsafe transfer maps without partially modifying a model.

### High

1. LaplacianConv used a full-strength edge residual at initialization. It now starts at zero and is bounded with `tanh`.
2. FFAFusion was inserted at every neck merge in FFA candidates. Each current FFA composite uses one P3 top-down fusion node, reducing optimization interference and making the module attributable.
3. CARAFE introduced an extra channel-reduction path and random upsampling behavior. It now preserves the intended channel contract and starts as exact nearest-neighbor upsampling before learning the content-aware residual.
4. Exact model YAMLs could be shadowed by a stale unified YAML name. Exact paths now take precedence.
5. Runtime activation resolution still used `eval`. It was replaced with a no-argument `torch.nn` activation allowlist; task inference now uses explicit attribute traversal.
6. SPDConv previously required two YAML layers and did not define odd-size behavior. It is now one atomic layer with deterministic padding and `pixel_unshuffle`, which preserves downstream layer numbering.

### Residual Risks

- A08 and A09 intentionally replace much of the neck. Their neck parameter transfer is only `58.347%` and `61.843%`; this is architectural novelty, not a remapping defect, but it makes them higher-risk experiments.
- P2 models inherit `86.562%` of Detect parameters because the new P2 branch has no baseline counterpart. This is expected and is now measured honestly.
- Local Windows can load duplicate OpenMP runtimes if `torch` is imported before the editable Ultralytics package. The audit uses the stable import order and no unsafe `KMP_DUPLICATE_LIB_OK` workaround. The remote Linux training environment is the authoritative runtime.
- PyTorch `.pt` checkpoints are trusted binary artifacts. Tier B now accepts only the recorded SHA256 of project-root `yolo26n.pt`; other workflows must still treat arbitrary `.pt` files as unsafe pickle input.
- Parameter counts in the audit are from default `nc=80` construction. Japan7 training rebuilds the head for `nc=7`, so remote summaries are lower. This is not evidence of a stale model.
- Buildability, identity initialization, and gradient stability do not establish an accuracy gain. Corrected models still require fresh, protocol-matched experiments.

## Adversarial Evidence

`scripts/audit_paper1_tiera_models.py` verified:

- safe YAML loading and exact YAML-path resolution;
- all 12 model constructions and finite full-model forward/backward passes;
- fused versus unfused inference consistency;
- exact identity initialization for attention/fusion modules and Conv-equivalence for LaplacianConv/FDConv;
- odd spatial sizes for SPDConv and CARAFE;
- CPU bfloat16 execution for FDConv;
- concurrent module re-entry and eight simultaneous reservations of the same run name;
- rejection of malicious activation strings, path traversal, unsafe prefix maps, and duplicate target mappings;
- atomic failure recovery: a rejected transfer leaves the destination model unchanged;
- parameter-weighted transfer coverage by backbone, neck, and Detect region.

Machine-readable evidence is stored in `experiments/module_scan/paper1_tiera_adversarial_audit.csv`; the readable summary is `experiments/module_scan/paper1_tiera_adversarial_audit.md`.

## Experimental Consequence

The 2026-07-11 A05 run (`mAP50=0.526`, `mAP50-95=0.295`) belongs to the pre-correction implementation at commit `be61dc3`. It remains valid evidence that the old design underperformed, but it must not be presented as a result for the corrected A05. All corrected runs need a new unique name and must keep `--checkpoint-remap auto`; manual layer-prefix remaps are rejected for semantic-map YAMLs.
