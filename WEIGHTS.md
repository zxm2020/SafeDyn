# WEIGHTS

SafeDyn core does **not** require learned weights.

The certificate chain (CertifiedAccept, exact rollout, backup validator, CBF-QP, etc.)
is a deterministic system with configurable parameters (tube margins, fallback priorities).
No neural network weights are needed.

## Optional: ETPNav

If you want to use SafeDyn with the ETPNav navigation policy:

1. Download the ETPNav checkpoint:
```bash
pip install modelscope
modelscope download --model admagic/ETPNav --local_dir ./data
```

2. The checkpoint should be at `./data/ETPNav.pt` or `./data/model_step_82500.pt`.
   - The fully fine-tuned checkpoint (required for paper-level navigation success)
     may differ from the warm-start checkpoint on ModelScope.

3. ETPNav also requires:
   - CLIP ViT-B/32 (auto-downloaded by transformers)
   - DDPO depth encoder (auto-downloaded)
   - BERT/LXMERT weights (auto-downloaded by HuggingFace)

## Without ETPNav Weights

Method demos and tests work without any weights. The `examples/` directory
contains self-contained Python scripts that exercise the certificate chain
with synthetic data.
