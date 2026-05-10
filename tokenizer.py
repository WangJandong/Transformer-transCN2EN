"""
BPE tokenizer built on SentencePiece for Chinese→English translation.
"""
import sentencepiece as spm
from pathlib import Path


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


def train_spm(
    input_files: list[str],
    model_prefix: str,
    vocab_size: int = 32000,
    character_coverage: float = 0.9995,
    model_type: str = "bpe",
    max_sentence_length: int = 256,
) -> None:
    """Train a SentencePiece BPE model on the combined source+target corpus."""
    spm.SentencePieceTrainer.train(
        input=input_files,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        max_sentence_length=max_sentence_length * 2,
        pad_id=PAD_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        unk_id=UNK_ID,
        pad_piece="<pad>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        unk_piece="<unk>",
        user_defined_symbols=[],
        num_threads=8,
        train_extremely_large_corpus=True,
        input_sentence_size=10_000_000,   # sample from the full set for speed
        shuffle_input_sentence=True,
    )


def load_spm(model_prefix: str) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    return sp


def build_tokenizer_if_needed(config) -> spm.SentencePieceProcessor:
    """Train the tokenizer if model files don't exist, then load it."""
    prefix = config.sp_model_prefix
    model_file = Path(f"{prefix}.model")
    if model_file.exists():
        print(f"Tokenizer already exists: {model_file}")
        return load_spm(prefix)

    print("Training SentencePiece tokenizer …")
    src_file = config.data_dir / f"train.{config.src_lang}"
    tgt_file = config.data_dir / f"train.{config.tgt_lang}"
    train_spm(
        input_files=[str(src_file), str(tgt_file)],
        model_prefix=prefix,
        vocab_size=config.vocab_size,
        character_coverage=config.character_coverage,
        model_type=config.sp_model_type,
        max_sentence_length=config.max_seq_len,
    )
    return load_spm(prefix)
