"""CPU checks for stable_flow.py: warm start, exact identity in the measuring pass, bounded logits.

Run: PCV_CODE_BASE=<code_base_v4> python test_stable_flow.py
"""
import copy
import os
from pathlib import Path
import sys

import torch

CODE_BASE = Path(os.environ.get('PCV_CODE_BASE', Path(__file__).resolve().parent / 'code_base_v4'))
sys.path.insert(0, str(CODE_BASE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from codec_flow_research import load_flow_state, flow_state  # noqa: E402
from small_object_flow import SmallObjectFlow  # noqa: E402
from stable_flow import StableObjectFlow, calibrate_gains, warm_start_from_small  # noqa: E402


def main():
    torch.manual_seed(0)
    objects, latent, enc, width, depth, heads = 12, 16, 24, 48, 2, 4
    anchors = torch.rand(objects, 2) * 2 - 1
    plain = SmallObjectFlow(objects, latent, enc, anchors, width, depth, heads)
    with torch.no_grad():
        plain.out.weight.normal_(std=.05)  # zero-init output would hide every difference
    stable = StableObjectFlow(objects, latent, enc, anchors, width, depth, heads)
    new = warm_start_from_small(stable, flow_state(plain))
    assert len(new) == 4 * depth, new  # one q and one k gain for self and for cross attention, per block
    z, t, ctx = torch.randn(3, objects, latent), torch.rand(3), torch.randn(3, 7, enc)
    expected = plain(z, t, ctx)
    measured, gains = calibrate_gains(stable, z, t, ctx)
    err = float((measured - expected).abs().max())
    assert err < 1e-4, f'measuring pass differs from SmallObjectFlow: {err}'
    assert all(g > 0 for g in gains)
    after = stable(z, t, ctx)
    assert torch.isfinite(after).all()
    # Logits stay bounded even if the projections blow up by 1000x.
    blown = copy.deepcopy(stable)
    with torch.no_grad():
        for block in blown.blocks:
            block.attn.in_proj_weight.mul_(1000)
    assert torch.isfinite(blown(z, t, ctx)).all()
    # Round trip through the checkpoint format.
    again = StableObjectFlow(objects, latent, enc, anchors, width, depth, heads)
    load_flow_state(again, flow_state(stable))
    assert torch.equal(again(z, t, ctx), stable(z, t, ctx))
    ema = copy.deepcopy(stable).requires_grad_(False)
    ema_state = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in ema.state_dict().items()}
    restored = StableObjectFlow(objects, latent, enc, anchors, width, depth, heads)
    restored.load_state_dict(ema_state, strict=True)
    assert torch.equal(restored(z, t, ctx), stable(z, t, ctx))
    print({'identity_max_abs_error': err, 'new_parameters': len(new), 'gains_example': gains[:4]})


if __name__ == '__main__':
    main()
