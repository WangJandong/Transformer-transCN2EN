"""
Transformer Encoder-Decoder for NMT with learned positional embeddings.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, E)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.embedding(positions)


class TranslationTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        activation: str = "relu",
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.src_pos = LearnedPositionalEmbedding(max_seq_len, d_model)
        self.tgt_pos = LearnedPositionalEmbedding(max_seq_len, d_model)

        self.embed_scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,  # pre-norm for training stability
        )

        self.output_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.full((sz, sz), float("-inf"), device=device), diagonal=1
        )

    def _create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S) → (B, S) with True where padding
        return x == self.pad_idx

    def forward(
        self,
        src_ids: torch.Tensor,      # (B, S_src)
        tgt_ids: torch.Tensor,      # (B, S_tgt)
    ) -> torch.Tensor:
        """Teacher-forcing forward pass. Returns logits (B, S_tgt, V)."""
        src_pad_mask = self._create_padding_mask(src_ids)
        tgt_pad_mask = self._create_padding_mask(tgt_ids)
        tgt_causal_mask = self._create_causal_mask(tgt_ids.size(1), tgt_ids.device)

        # tgt_mask for nn.Transformer must be (S_tgt, S_tgt) for batch_first=True it's still (T, T)
        src_emb = self.dropout(self.src_pos(self.src_embed(src_ids) * self.embed_scale))
        tgt_emb = self.dropout(self.tgt_pos(self.tgt_embed(tgt_ids) * self.embed_scale))

        memory = self.transformer.encoder(
            src=src_emb,
            src_key_padding_mask=src_pad_mask,
        )

        output = self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )

        return self.output_proj(output)

    @torch.no_grad()
    def translate(
        self,
        src_ids: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 256,
        beam_size: int = 1,
    ) -> torch.Tensor:
        """Greedy or beam-search decode. Returns (B, S)."""
        if beam_size == 1:
            return self._greedy_decode(src_ids, bos_id, eos_id, max_len)
        return self._beam_search_decode(src_ids, bos_id, eos_id, max_len, beam_size)

    def _greedy_decode(
        self, src_ids: torch.Tensor, bos_id: int, eos_id: int, max_len: int
    ) -> torch.Tensor:
        B = src_ids.size(0)
        device = src_ids.device
        rep_penalty = 1.2  # discourage repeating the same token

        src_pad_mask = self._create_padding_mask(src_ids)
        src_emb = self.dropout(self.src_pos(self.src_embed(src_ids) * self.embed_scale))
        memory = self.transformer.encoder(src=src_emb, src_key_padding_mask=src_pad_mask)

        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_emb = self.dropout(self.tgt_pos(self.tgt_embed(ys) * self.embed_scale))
            tgt_causal_mask = self._create_causal_mask(ys.size(1), device)

            out = self.transformer.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_causal_mask,
                memory_key_padding_mask=src_pad_mask,
            )
            logits = self.output_proj(out[:, -1, :])  # (B, V)
            # repetition penalty: penalize tokens already generated
            if rep_penalty != 1.0:
                for b in range(B):
                    for t in ys[b]:
                        if t != bos_id and t != eos_id:
                            logits[b, t] /= rep_penalty
            next_token = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            ys = torch.cat([ys, next_token], dim=-1)

            just_finished = (next_token.squeeze(-1) == eos_id)
            finished = finished | just_finished
            if finished.all():
                break

        return ys

    def _beam_search_decode(
        self,
        src_ids: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
        beam_size: int,
    ) -> torch.Tensor:
        """Simple beam search for a single sentence (B=1)."""
        assert src_ids.size(0) == 1, "Beam search currently only supports batch_size=1"
        device = src_ids.device

        src_pad_mask = self._create_padding_mask(src_ids)
        src_emb = self.dropout(self.src_pos(self.src_embed(src_ids) * self.embed_scale))
        memory = self.transformer.encoder(src=src_emb, src_key_padding_mask=src_pad_mask)
        memory = memory.expand(beam_size, -1, -1)
        src_pad_mask = src_pad_mask.expand(beam_size, -1)

        ys = torch.full((1, 1), bos_id, dtype=torch.long, device=device).expand(beam_size, -1)
        scores = torch.zeros(beam_size, device=device)  # cumulative log-prob

        for step in range(max_len):
            tgt_emb = self.dropout(self.tgt_pos(self.tgt_embed(ys) * self.embed_scale))
            tgt_causal_mask = self._create_causal_mask(ys.size(1), device)

            out = self.transformer.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_causal_mask,
                memory_key_padding_mask=src_pad_mask,
            )
            logits = self.output_proj(out[:, -1, :])  # (B*mem, V)
            log_probs = F.log_softmax(logits, dim=-1)
            cand_scores = scores.unsqueeze(-1) + log_probs  # (B*mem, V)

            if step == 0:
                topk_scores, topk_idx = cand_scores[0].topk(beam_size)  # first step only
                ys = torch.cat([ys, topk_idx.unsqueeze(-1)], dim=-1)
                scores = topk_scores
            else:
                topk_scores, topk_idx = cand_scores.view(-1).topk(beam_size)
                beam_indices = topk_idx // log_probs.size(-1)
                token_indices = topk_idx % log_probs.size(-1)
                ys = torch.cat(
                    [ys[beam_indices], token_indices.unsqueeze(-1)], dim=-1
                )
                scores = topk_scores
                memory = memory[beam_indices]
                src_pad_mask = src_pad_mask[beam_indices]

            if (ys[:, -1] == eos_id).all():
                break

        best = ys[0]  # highest score beam
        return best.unsqueeze(0)
