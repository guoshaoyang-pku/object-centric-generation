"""Direct object-token access to an already loaded Wan 0.35.2 backbone.

This is an opt-in research adapter. It does not load weights or replace any
parameters in the supplied ``WanObjectFlow``. The pretrained blocks, existing
LoRA modules and input/output/condition adapters are shared by reference.

Integration, before constructing the optimizer or DDP wrapper::

    model.flow = WanTokenAdapter.from_legacy(
        model.flow, position_mode="object_xy", object_anchors=model.decoder.anchors[..., :2])

``object_xy`` uses explicit codec anchor x/y coordinates and no fake temporal
grid. It preserves the anchor-conditioned slot semantics of the current codec.
``none`` removes every object position signal and accepts any positive token
count, but is suitable only when the codec/target distribution is exchangeable.
``legacy_grid`` retains the original RoPE and is the numerical-equivalence
control. The posterior and renderer elsewhere in the model may still require
250 objects; changing this adapter alone does NOT make those variable-length.

The default forward retains the factorized projections and their Parameter
identities. ``fused_projection_tensors`` is a detached inference/export utility:
training a single dense fused layer would change the trainable parameter space
and would not preserve the original optimizer state or gradient update.

The block plumbing follows the pinned Diffusers v0.35.2 transformer_wan.py.
Validate against the actual loaded model in legacy_grid mode before GPU use.
"""
import math
import hashlib
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


def patch_projection_weight(backbone):
    """Conv3d weight in the project's [pt, ph, pw, channel] token order.

    Flattening Conv3d.weight directly is wrong: it stores channel before the
    kernel dimensions. Conv stride==kernel makes each patch independent.
    """
    patch = backbone.patch_embedding
    if not isinstance(patch, nn.Conv3d):
        raise TypeError("Expected Wan Conv3d patch_embedding")
    kernel = tuple(patch.kernel_size)
    if tuple(patch.stride) != kernel or tuple(patch.padding) != (0, 0, 0):
        raise ValueError("Patch projection requires non-overlapping unpadded patches")
    if tuple(patch.dilation) != (1, 1, 1) or patch.groups != 1:
        raise ValueError("Unsupported patch dilation or groups")
    if tuple(backbone.config.patch_size) != kernel:
        raise ValueError("Patch config and Conv3d kernel disagree")
    return patch.weight.permute(0, 2, 3, 4, 1).reshape(patch.out_channels, -1)


def xy_position_features(anchors, width):
    """Fixed Fourier features of real codec xy anchors; no slot-number/time axis."""
    if anchors.ndim != 2 or anchors.shape[1] != 2 or width < 1:
        raise ValueError("Expected [objects,2] xy anchors and positive hidden width")
    bands = max(1, math.ceil(width / 4))
    frequency = torch.exp(torch.linspace(0, math.log(16), bands,
                                        device=anchors.device, dtype=torch.float32)) * math.pi
    phase = anchors.float()[..., None] * frequency
    return torch.cat((phase.sin(), phase.cos()), -1).flatten(-2)[..., :width]


def anchor_contract(anchors, scale):
    cpu = anchors.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return {"shape": list(cpu.shape), "float32_sha256": hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
            "feature_version": "xy-fourier-exp1to16-pi-v1", "feature_scale": float(scale)}


@torch.no_grad()
def fused_projection_tensors(flow, dtype=torch.float32):
    """Export composed affine maps without mutating modules or parameters.

    The composition is algebraically exact over real numbers. Floating-point
    rounding differs from the two sequential projections, especially in BF16.
    Returned tensors are detached and remain on their source device.
    """
    patch = flow.backbone.patch_embedding
    wp = patch_projection_weight(flow.backbone).to(dtype=dtype)
    wi = flow.input.weight.to(dtype=dtype)
    bi = flow.input.bias.to(dtype=dtype) if flow.input.bias is not None else wi.new_zeros(wi.shape[0])
    bp = patch.bias.to(dtype=dtype) if patch.bias is not None else wp.new_zeros(wp.shape[0])
    head = flow.backbone.proj_out
    wh = head.weight.to(dtype=dtype)
    bh = head.bias.to(dtype=dtype) if head.bias is not None else wh.new_zeros(wh.shape[0])
    wo = flow.output.weight.to(dtype=dtype)
    bo = flow.output.bias.to(dtype=dtype) if flow.output.bias is not None else wo.new_zeros(wo.shape[0])
    return {"input_weight": (wp @ wi).detach(), "input_bias": (wp @ bi + bp).detach(),
            "output_weight": (wo @ wh).detach(), "output_bias": (wo @ bh + bo).detach()}


class WanTokenAdapter(nn.Module):
    """The WanObjectFlow API with direct tokens and explicit position policy."""

    def __init__(self, legacy, position_mode="none", object_anchors=None, anchor_scale=.1):
        super().__init__()
        if position_mode not in {"none", "legacy_grid", "object_xy"}:
            raise ValueError("position_mode must be 'none', 'legacy_grid' or 'object_xy'")
        for name in ("backbone", "input", "output", "condition", "grid", "patch_size"):
            if not hasattr(legacy, name):
                raise TypeError("Expected a WanObjectFlow-compatible module: missing " + name)
        self.backbone = legacy.backbone
        self.input = legacy.input
        self.output = legacy.output
        self.condition = legacy.condition
        self.grid = tuple(legacy.grid)  # Used only for the equivalence control.
        self.patch_size = tuple(legacy.patch_size)
        self.channels = int(self.backbone.config.in_channels)
        self.position_mode = position_mode
        self.lora_modules = list(getattr(legacy, "lora_modules", []))
        required = ("rope", "blocks", "condition_embedder", "norm_out", "proj_out", "scale_shift_table")
        if any(not hasattr(self.backbone, key) for key in required):
            raise TypeError("Expected the Diffusers 0.35.2 Wan backbone interface")
        wp = patch_projection_weight(self.backbone)
        if self.input.out_features != wp.shape[1]:
            raise ValueError("Input adapter and patch embedding dimensions disagree")
        if self.output.in_features != self.backbone.proj_out.out_features:
            raise ValueError("Output adapter and pretrained head dimensions disagree")
        if self.input.in_features != self.output.out_features:
            raise ValueError("Object latent input and output widths must match")
        self.anchor_scale = float(anchor_scale)
        if position_mode == "object_xy":
            if not isinstance(object_anchors, torch.Tensor) or object_anchors.ndim != 2 or object_anchors.shape[1] != 2:
                raise ValueError("object_xy requires explicitly supplied [objects,2] codec anchors")
            if object_anchors.shape[0] < 1 or not bool(torch.isfinite(object_anchors).all()):
                raise ValueError("object_xy anchors must be nonempty and finite")
            if not math.isfinite(self.anchor_scale) or self.anchor_scale <= 0:
                raise ValueError("object_xy anchor_scale must be finite and positive")
            anchors = object_anchors.detach().to(device=self.backbone.patch_embedding.weight.device,
                                                 dtype=torch.float32).clone()
            self.register_buffer("object_anchors", anchors)
            # Keep construction-time metadata independent of the mutable buffer:
            # load_state_dict copies buffers before invoking set_extra_state.
            self._expected_anchor_contract = anchor_contract(anchors, self.anchor_scale)
        else:
            if object_anchors is not None:
                raise ValueError("Explicit anchors are only accepted with position_mode='object_xy'")
            self.register_buffer("object_anchors", None)
            self._expected_anchor_contract = None
        # Wrapping a running/eval object should not silently reset module mode.
        self.training = legacy.training

    @classmethod
    def from_legacy(cls, legacy, position_mode="none", object_anchors=None, anchor_scale=.1):
        return cls(legacy, position_mode=position_mode, object_anchors=object_anchors, anchor_scale=anchor_scale)

    def get_extra_state(self):
        # This prevents state-only checkpoint loaders silently restoring the
        # wrong positional policy. Existing filtered checkpoint writers must
        # explicitly save this field or record policy in their run config.
        return {"version": 2, "position_mode": self.position_mode, "legacy_grid": self.grid,
                "object_anchor_contract": self._expected_anchor_contract}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Checkpoint token-adapter position policy differs from the constructed model")
        if self.object_anchors is not None and anchor_contract(self.object_anchors, self.anchor_scale) != self._expected_anchor_contract:
            raise ValueError("Checkpoint object anchor buffer differs from its recorded contract")

    def load_legacy_state_dict(self, state_dict):
        """Strictly import legacy flow weights, explicitly adding new metadata."""
        state = OrderedDict(state_dict)
        if "_extra_state" in state:
            raise ValueError("Expected legacy state without token-adapter metadata")
        state["_extra_state"] = self.get_extra_state()
        if self.object_anchors is not None:
            state["object_anchors"] = self.object_anchors.detach().clone()
        return self.load_state_dict(state, strict=True)

    def _position(self, z):
        if self.position_mode in {"none", "object_xy"}:
            return None
        if z.shape[1] != math.prod(self.grid):
            raise ValueError("legacy_grid requires its original token count")
        # WanRotaryPosEmbed 0.35.2 reads shape only; expanded view allocates no
        # dense fake video. No token data is sent through this view.
        shape = tuple(g * p for g, p in zip(self.grid, self.patch_size))
        shape_only = z.new_empty(1).expand(z.shape[0], self.channels, *shape)
        return self.backbone.rope(shape_only)

    def forward(self, z, t, context):
        if z.ndim != 3 or z.shape[1] <= 0 or z.shape[2] != self.input.in_features:
            raise ValueError("z must be [batch, positive object count, latent_dim]")
        if context.ndim != 3 or context.shape[0] != z.shape[0]:
            raise ValueError("context must be [batch, condition count, feature width]")
        if tuple(t.shape) not in {(z.shape[0],), (z.shape[0], z.shape[1])}:
            raise ValueError("t must be [batch] or [batch, object count]")
        if self.position_mode == "object_xy" and z.shape[1] != self.object_anchors.shape[0]:
            raise ValueError("object_xy token count must equal the explicitly supplied codec anchor count")
        with torch.autocast(device_type=z.device.type, dtype=torch.bfloat16,
                            enabled=z.device.type == "cuda"):
            hidden = F.linear(self.input(z), patch_projection_weight(self.backbone),
                              self.backbone.patch_embedding.bias)
            if self.position_mode == "object_xy":
                position = xy_position_features(self.object_anchors, hidden.shape[-1])
                hidden = hidden + self.anchor_scale * position[None].to(dtype=hidden.dtype)
            condition = self.condition(context)
            rotary = self._position(z)
            # Same noise(0)->clean(1) convention as the existing object FM.
            native_t = (1 - t.float()) * 1000
            seq_len = native_t.shape[1] if native_t.ndim == 2 else None
            if seq_len is not None:
                native_t = native_t.flatten()
            temb, modulation, condition, image_condition = self.backbone.condition_embedder(
                native_t, condition, encoder_hidden_states_image=None, timestep_seq_len=seq_len)
            modulation = modulation.unflatten(2 if seq_len is not None else 1, (6, -1))
            if image_condition is not None:
                condition = torch.cat([image_condition, condition], dim=1)
            use_checkpoint = torch.is_grad_enabled() and bool(getattr(self.backbone, "gradient_checkpointing", False))
            for block in self.backbone.blocks:
                if use_checkpoint:
                    hidden = self.backbone._gradient_checkpointing_func(block, hidden, condition, modulation, rotary)
                else:
                    hidden = block(hidden, condition, modulation, rotary)
            if temb.ndim == 3:
                shift, scale = (self.backbone.scale_shift_table.unsqueeze(0).to(temb.device)
                                + temb.unsqueeze(2)).chunk(2, dim=2)
                shift, scale = shift.squeeze(2), scale.squeeze(2)
            else:
                shift, scale = (self.backbone.scale_shift_table.to(temb.device)
                                + temb.unsqueeze(1)).chunk(2, dim=1)
            hidden = (self.backbone.norm_out(hidden.float()) * (1 + scale.to(hidden.device))
                      + shift.to(hidden.device)).type_as(hidden)
            return self.output(self.backbone.proj_out(hidden))
