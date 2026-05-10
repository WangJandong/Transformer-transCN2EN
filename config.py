from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # data
    data_dir: Path = Path("data")
    src_lang: str = "zh"
    tgt_lang: str = "en"
    max_train_samples: int = 0           # 0 = all available training pairs

    # tokenizer
    sp_model_prefix: str = "spm_bpe"
    vocab_size: int = 32000
    character_coverage: float = 0.9995
    sp_model_type: str = "bpe"  # bpe or unigram

    # model
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 96           # p99 < 63 tok; 96 covers all with VRAM headroom
    activation: str = "relu"

    # training
    batch_size: int = 128          # 22GB VRAM: stable at max_seq_len=128
    epochs: int = 20
    lr: float = 1.0                # factor for the Noam scheduler
    warmup_steps: int = 4000
    label_smoothing: float = 0.1
    grad_accum_steps: int = 2
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    fused_adamw: bool = True       # fused CUDA AdamW kernel, ~20% optimizer speedup

    # torch.compile — Turing SM 7.5 optimizations
    # "default"        — basic fusion, fast compile, 10-15% uplift
    # "reduce-overhead" — CUDA graphs, less kernel launch overhead, 15-25% uplift
    # "max-autotune"    — autotune matmul shapes, best perf but slow first compile
    # None / ""         — disabled
    compile_mode: str = ""              # inductor crashes on dynamic shapes, cudagraphs conflicts with GradScaler

    # system
    num_workers: int = 2                # pre-tokenized data needs fewer workers
    seed: int = 42
    log_interval: int = 500             # ~every 1 min
    save_interval: int = 10000          # ~every 15 min
    val_interval: int = 5000            # ~every 8 min
    checkpoint_dir: Path = Path("checkpoints")
    device: str = "cuda"
