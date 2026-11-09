"""A GPT-2 style decoder trained from scratch. Requires PyTorch 2.x.

Reviewed against GPT-2 small (Claude): pre-LN blocks, tanh GELU, learned
positions, biases, final LayerNorm, tied input/output embeddings, N(0, 0.02)
init with residual projections scaled by 1/sqrt(2 * n_layers). The loss is
computed in chunks so the T4 doesn't run out of memory (chunked_cross_entropy).
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

# Rows of the (B*T, vocab) logits handled per loss chunk. Each chunk's FP32
# softmax is ~2048 * 50257 * 4 bytes = 0.4 GB, recomputed in backward.
LOSS_CHUNK_TOKENS = 2048


IGNORE_INDEX = -100


def _cross_entropy_sum(logits, targets):
    # FP32 softmax for this chunk only; works the same under autocast.
    # ignore_index lets SFT mask prompt tokens (completion-only loss).
    return F.cross_entropy(
        logits.float(),
        targets,
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )


def chunked_cross_entropy(logits, targets, chunk_tokens=LOSS_CHUNK_TOKENS):
    """Mean cross-entropy over supervised positions, computed in row chunks.

    The plain `F.cross_entropy(logits.float(), ...)` materialises an FP32
    copy of the full logits plus its FP32 log-softmax: at batch 12 x 1024 x
    50257 that is ~2.5 GB each, on top of the FP16 logits, which is what
    OOM'd the T4. Here each chunk's FP32 work is wrapped in activation
    checkpointing, so autograd keeps only the (already existing) low-precision
    logits and recomputes one chunk's softmax at a time in backward. Same
    value and gradients as the plain loss; the recompute is a cheap,
    memory-bound softmax, not the vocabulary matmul.

    Targets equal to IGNORE_INDEX (-100) are excluded from the mean, which
    enables completion-only SFT without changing the pretraining path.
    """
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    # Tensor ops only (no .item() / Python if on counts) so torch.compile(fullgraph=True) works.
    # Suggested by Claude's Stage E mask audit; applied as Lumi integration glue.
    count = (flat_targets != IGNORE_INDEX).sum()
    use_checkpoint = torch.is_grad_enabled() and flat_logits.requires_grad
    total = flat_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, flat_targets.numel(), chunk_tokens):
        rows = slice(start, start + chunk_tokens)
        chunk_targets = flat_targets[rows]
        if use_checkpoint:
            part = checkpoint(_cross_entropy_sum, flat_logits[rows], chunk_targets, use_reentrant=False)
        else:
            part = _cross_entropy_sum(flat_logits[rows], chunk_targets)
        total = total + part
    # Fully-ignored batch: clamp keeps a connected zero (sum of ignored CE is 0).
    return total / count.clamp(min=1).to(total.dtype)


def _save_tied_weight(module, state_dict, prefix, local_metadata):
    # Keep only the embedding alias so safetensors.save_file accepts the state.
    state_dict.pop(prefix + "lm_head.weight", None)


def _load_tied_weight(module, state_dict, prefix, local_metadata, strict,
                      missing_keys, unexpected_keys, error_msgs):
    key = prefix + "token_embedding.weight"
    if key in state_dict:
        state_dict[prefix + "lm_head.weight"] = state_dict[key]


def _retie_weight(module, incompatible_keys):
    # assign=True can replace the two Parameters independently during loading.
    module.lm_head.weight = module.token_embedding.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (
            t.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
            for t in (q, k, v)
        )
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(batch, length, width)
        return self.resid_dropout(self.c_proj(out))


class MLP(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class PlaygroundLM(nn.Module):
    def __init__(self, vocab_size=50257, n_layers=12, n_heads=12,
                 d_model=768, context_len=1024, dropout=0.0, **kwargs):
        super().__init__()
        if min(vocab_size, n_layers, n_heads, d_model, context_len) <= 0:
            raise ValueError("Model dimensions must be positive")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.context_len = context_len
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight
        for block in self.blocks:
            for projection in (block.attn.c_proj, block.mlp.c_proj):
                nn.init.normal_(projection.weight, std=0.02 / math.sqrt(2 * n_layers))
        self.register_state_dict_post_hook(_save_tied_weight)
        self.register_load_state_dict_pre_hook(_load_tied_weight)
        self.register_load_state_dict_post_hook(_retie_weight)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x, y=None):
        if x.ndim != 2 or not 0 < x.shape[1] <= self.context_len:
            raise ValueError("x must have shape (batch, T), 1 <= T <= context_len")
        if y is not None and y.shape != x.shape:
            raise ValueError("y must have the same shape as x")
        positions = torch.arange(x.shape[1], device=x.device)
        hidden = self.dropout(
            self.token_embedding(x) + self.position_embedding(positions)
        )
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.ln_f(hidden))
        loss = None
        if y is not None:
            # FP32 cross-entropy, chunked so it never holds a full FP32 copy
            # of the B x T x vocab logits (see chunked_cross_entropy).
            loss = chunked_cross_entropy(logits, y, LOSS_CHUNK_TOKENS)
        return logits, loss

    def num_params(self, non_embedding=True):
        """Count unique parameters; by default exclude position embeddings only.

        Token embeddings also serve as the output head, so they remain counted.
        """
        count = sum(p.numel() for p in self.parameters())
        if non_embedding:
            count -= self.position_embedding.weight.numel()
        return count

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Sample tokens with a sliding context window; preserve training mode."""
        if idx.ndim != 2 or idx.shape[1] == 0:
            raise ValueError("idx must be a nonempty (batch, T) prompt")
        if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            raise ValueError("top_k must be a positive integer or None")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -self.context_len:])
                logits = logits[:, -1, :].float() / temperature
                if top_k is not None:
                    values, indices = torch.topk(
                        logits, min(top_k, self.vocab_size), dim=-1
                    )
                    logits = torch.full_like(logits, float("-inf")).scatter(
                        -1, indices, values
                    )
                next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
                idx = torch.cat((idx, next_token), dim=1)
        finally:
            self.train(was_training)
        return idx
