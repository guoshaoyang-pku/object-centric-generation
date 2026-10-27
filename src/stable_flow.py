"""Small object flow with per-head QK RMS normalization.

All four 10-27 runs diverged after 115k-220k optimizer steps at a constant LR
of 1e-4. Before the blow-up the first block's attention projection and the
time-embedding MLP grew steadily (time_mlp.0 weight norm 10.6 -> 19.6); then the
attention logits exploded, gradients grew without bound and the flow collapsed
to a constant prediction (loss about 1.62). Normalizing q and k per head bounds
every attention logit by sqrt(head_dim) times a learned per-dimension gain.

Every other module, parameter name and shape is the one in SmallObjectFlow, so
an existing SmallObjectFlow state loads unchanged; only the gains are new
(initialized to 1). The function changes, because the old logits were not
normalized, so a warm start is an initialization, not an equivalence.
"""
import torch
from torch import nn
from torch.nn import functional as F

from model import FlowBlock
from small_object_flow import SmallObjectFlow


class QKNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.measure = None  # a list while calibrating: pass x through, record its RMS

    def forward(self, x):
        if self.measure is not None:
            self.measure.append(float(x.float().square().mean(-1).sqrt().mean()))
            return x
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + self.eps) * self.gain).to(x.dtype)


def qk_attention(mha, q_norm, k_norm, query, key):
    """The weights of an nn.MultiheadAttention, with q and k RMS-normalized per head."""
    b, n, width = query.shape
    m, heads = key.shape[1], mha.num_heads
    d = width // heads
    wq, wk, wv = mha.in_proj_weight.chunk(3)
    bq, bk, bv = mha.in_proj_bias.chunk(3)
    q = F.linear(query, wq, bq).view(b, n, heads, d).transpose(1, 2)
    k = F.linear(key, wk, bk).view(b, m, heads, d).transpose(1, 2)
    v = F.linear(key, wv, bv).view(b, m, heads, d).transpose(1, 2)
    out = F.scaled_dot_product_attention(q_norm(q), k_norm(k), v)
    return mha.out_proj(out.transpose(1, 2).reshape(b, n, width))


class StableFlowBlock(FlowBlock):
    """FlowBlock.forward with QK-normalized self and cross attention."""

    def forward(self, x, context, t):
        a, b, c = self.time(F.silu(t)).chunk(3, -1)
        h = self.norm1(x) + a[:, None]
        x = x + qk_attention(self.attn, self.q_self, self.k_self, h, h)
        h = self.norm2(x) + b[:, None]
        x = x + qk_attention(self.cross, self.q_cross, self.k_cross, h, context)
        return x + self.ff(self.norm3(x) + c[:, None])


class StableObjectFlow(SmallObjectFlow):
    def __init__(self, objects, latent_dim, encoder_width, anchors, width=384, depth=8, heads=6):
        super().__init__(objects, latent_dim, encoder_width, anchors, width, depth, heads)
        head_dim = width // heads
        for block in self.blocks:
            if type(block) is not FlowBlock or block.attn.dropout or block.cross.dropout:
                raise ValueError('Expected the plain, dropout-free FlowBlock')
            for name in ['q_self', 'k_self', 'q_cross', 'k_cross']:
                block.add_module(name, QKNorm(head_dim))
            block.__class__ = StableFlowBlock

    def get_extra_state(self):
        return {**super().get_extra_state(), 'qk_norm': 'per-head RMS on q and k, learned gain, eps 1e-6'}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('Stable-flow architecture or positional contract differs')


@torch.no_grad()
def calibrate_gains(flow, z, t, context):
    """Set each gain to the mean RMS its q or k had without normalization.

    During the measuring pass every QKNorm is the identity, so that pass computes
    exactly the warm-start SmallObjectFlow; its output is returned for checking.
    Afterwards typical tokens keep roughly their old logit scale.
    """
    norms = [m for m in flow.modules() if isinstance(m, QKNorm)]
    for m in norms:
        m.measure = []
    try:
        out = flow(z, t, context)
    finally:
        values = [sum(m.measure) / len(m.measure) for m in norms]
        for m in norms:
            m.measure = None
    for m, value in zip(norms, values):
        m.gain.fill_(value)
    return out, values


def warm_start_from_small(flow, state):
    """Load a SmallObjectFlow state into a StableObjectFlow; returns the new parameter names."""
    state = dict(state)
    state['_extra_state'] = flow.get_extra_state()
    result = flow.load_state_dict(state, strict=False)
    new = {n for n, _ in flow.named_parameters()
           if any(f'.{name}.' in n for name in ['q_self', 'k_self', 'q_cross', 'k_cross'])}
    if result.unexpected_keys or set(result.missing_keys) != new:
        raise ValueError(f'Warm start mismatch: unexpected {result.unexpected_keys[:4]}, '
                         f'missing {sorted(set(result.missing_keys) ^ new)[:4]}')
    return sorted(new)
