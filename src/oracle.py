"""Oracle object codec and flow (amendment F): GT instance masks assign pixels to parents.

Question: if each token is one whole object, tracked through time with
pixel-exact support, does prefix-conditioned flow generation of the future
improve (heldout MSE, motion, long-rollout drift) over the patch-grid codec?
This is an upper-bound probe; it does not discover objects.

Representation (250 parents, the same count, latent width and decoder family
as the C2 codec):
- 25 object slots. Prefix-visible instances take slots 0..m-1 in instance
  order; instances first seen in the future take the next slots in order of
  first appearance; the rest stay empty. Instances beyond 25 count as
  background (none in MOVi-E so far; counted per clip).
- 225 background parents on a fixed 15x15 anchor grid; each reads background
  pixels in a window around its anchor.
- Encoder: no VAE features and no patch tokens. For every parent and frame a
  16x16 nearest-neighbour glimpse of its own pixels (RGB times its mask, plus
  the mask) over its mask's bounding box, with the box geometry as features,
  then a temporal transformer over frames. A parent never sees another
  instance's pixels.
- Decoder: TrajectoryDecoder (Bezier center, affine, 4 child Gaussians, 109
  RBF textures, softmax density mixing), trained from scratch. Object slots
  have anchor (0, 0), so center = tanh(.) spans the frame; background anchors
  are the grid. A presence logit per object slot lets empty slots vanish.
- Training: RGB MSE of the normal mixture, MSE of the GT-routed mixture, and
  cross-entropy of the per-group mixture weight against the GT slot label.
  At render time no mask is used.

Leakage: the flow conditions on the 4 observed frames and their masks only.
In those frames only prefix-visible instances have pixels, so the condition
shows slots 0..m-1 and nothing about objects that appear later.
"""
import hashlib
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CODE_BASE = Path(os.environ.get('PCV_CODE_BASE', Path(__file__).resolve().parent / 'code_base_v4'))
sys.path.insert(0, str(CODE_BASE))
from codec_research import CacheBank  # noqa: E402
from trajectory_codec import TrajectoryDecoder  # noqa: E402
from stable_flow import StableObjectFlow  # noqa: E402

SLOTS = 25
BG_SIDE = 15
GLIMPSE = 16


def background_anchors(side=BG_SIDE):
    g = torch.linspace(-.9, .9, side)
    yy, xx = torch.meshgrid(g, g, indexing='ij')
    return torch.stack((xx, yy), -1).reshape(-1, 2)


def slot_labels(seg, slots=SLOTS, prefix=4):
    """seg [24,H,W] uint8 instance ids (0 = background) -> labels in 0..slots (slots = background)."""
    n = int(seg.max())
    counts = np.stack([np.bincount(f.ravel(), minlength=n + 1) for f in seg])  # [24, n+1]
    seen = counts[:prefix, 1:].sum(0) > 0
    later = (counts[:, 1:] > 0)
    first = np.where(later.any(0), later.argmax(0), 99)
    order = [i for i in range(n) if seen[i]] + sorted((i for i in range(n) if not seen[i] and first[i] < 99),
                                                      key=lambda i: (first[i], i))
    lut = np.full(256, slots, np.uint8)
    for slot, inst in enumerate(order[:slots]):
        lut[inst + 1] = slot
    return lut[seg], {'instances': n, 'prefix_visible': int(seen.sum()), 'used_slots': min(len(order), slots),
                      'overflow_instances': max(len(order) - slots, 0)}


class OracleBank(CacheBank):
    """CacheBank provenance plus GT slot labels. features()/batch() feed codec_flow_research.evaluate unchanged."""

    def __init__(self, data, cache, seg_dir, slots=SLOTS):
        super().__init__(data, cache)
        self.seg_dir, self.slots = Path(seg_dir), slots
        self.labels, self.slot_info, self.flip = {}, {}, False
        self.checksums = {}
        log = self.seg_dir / 'clips.jsonl'
        if log.exists():
            import json
            for line in log.read_text().splitlines():
                row = json.loads(line)
                if row.get('status') == 'complete':
                    self.checksums[row['id']] = row['npz_sha256']

    def label(self, identifier):
        if identifier not in self.labels:
            path = self.seg_dir / self.records[identifier]['split'] / (identifier + '.npz')
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != self.checksums.get(identifier):
                raise ValueError('Segmentation checksum mismatch: ' + identifier)
            import io
            with np.load(io.BytesIO(raw), allow_pickle=False) as f:
                seg = f['segmentations']
            self.labels[identifier], self.slot_info[identifier] = slot_labels(seg, self.slots)
        return self.labels[identifier]

    def preload(self, ids, workers=32):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(self.label, ids))

    def features(self, identifier):
        # Duck-typed for codec_flow_research.prefix_batch: observed frames and their masks only.
        return {'prefix': np.asarray(self.rgb(identifier)[:4]), 'prefix_coords': self.label(identifier)[:4]}

    def batch(self, ids, device):
        rgb = torch.from_numpy(np.stack([np.asarray(self.rgb(i)[4:]) for i in ids])).to(device)
        lab = torch.from_numpy(np.stack([self.label(i)[4:] for i in ids])).to(device)
        return rgb, lab, None

    def clip_tensors(self, ids, device, flips=None):
        """Full 24 frames: RGB uint8 [B,24,H,W,3] and labels uint8 [B,24,H,W], optional per-clip mirror."""
        rgb = torch.from_numpy(np.stack([np.asarray(self.rgb(i)) for i in ids])).to(device)
        lab = torch.from_numpy(np.stack([self.label(i) for i in ids])).to(device)
        if flips is not None and flips.any():
            f = flips.to(device)
            rgb = torch.where(f[:, None, None, None, None], rgb.flip(3), rgb)
            lab = torch.where(f[:, None, None, None], lab.flip(3), lab)
        return rgb, lab


def fourier(x, bands=4):
    freq = (2. ** torch.arange(bands, device=x.device, dtype=torch.float32)) * math.pi
    phase = x.float()[..., None] * freq
    return torch.cat((phase.sin(), phase.cos()), -1).flatten(-2)


class GlimpseEncoder(nn.Module):
    """Per-parent masked glimpses over frames -> one token per parent."""

    def __init__(self, slots=SLOTS, bg_side=BG_SIDE, width=256, glimpse=GLIMPSE, depth=2):
        super().__init__()
        self.slots, self.glimpse, self.width = slots, glimpse, width
        bg = background_anchors(bg_side)
        self.register_buffer('bg_centers', bg)
        self.bg_half = 1.5 * 1.8 / (bg_side - 1)
        self.parents = slots + len(bg)
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Flatten(), nn.Linear(128 * (glimpse // 4) ** 2, width))
        self.geometry = nn.Linear(4 * 8 + 2, width)
        self.frame = nn.Embedding(24, width)
        self.parent = nn.Parameter(torch.randn(1, self.parents, width) * .02)
        layer = nn.TransformerEncoderLayer(width, 4, 4 * width, dropout=0., activation='gelu',
                                           batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)

    def boxes(self, lab):
        """lab [B,T,H,W] long in 0..slots -> boxes [B,T,N,4] (cx,cy,hx,hy), area fraction and visibility [B,T,N]."""
        b, t, h, w = lab.shape
        s, dev = self.slots, lab.device
        flat = lab.reshape(b, t, h * w)
        xs = torch.linspace(-1, 1, w, device=dev).repeat(h).expand(b, t, -1)
        ys = torch.linspace(-1, 1, h, device=dev).repeat_interleave(w).expand(b, t, -1)
        area = torch.zeros(b, t, s + 1, device=dev).scatter_add_(2, flat, torch.ones_like(xs))
        lo = lambda v: torch.full((b, t, s + 1), 2., device=dev).scatter_reduce_(2, flat, v, 'amin')
        hi = lambda v: torch.full((b, t, s + 1), -2., device=dev).scatter_reduce_(2, flat, v, 'amax')
        x0, x1, y0, y1 = lo(xs)[..., :s], hi(xs)[..., :s], lo(ys)[..., :s], hi(ys)[..., :s]
        visible = area[..., :s] > 0
        px = 2. / (w - 1)
        cx = torch.where(visible, (x0 + x1) / 2, 0.)
        cy = torch.where(visible, (y0 + y1) / 2, 0.)
        hx = torch.where(visible, ((x1 - x0) / 2 + 2 * px).clamp(min=3 * px), 1.)
        hy = torch.where(visible, ((y1 - y0) / 2 + 2 * px).clamp(min=3 * px), 1.)
        obj = torch.stack((cx, cy, hx, hy), -1)
        bg = torch.cat((self.bg_centers, torch.full_like(self.bg_centers, self.bg_half)), -1)
        boxes = torch.cat((obj, bg.expand(b, t, -1, -1)), 2)
        # Background parents: fraction of their window that is background is not needed; visibility = 1.
        frac = torch.cat((area[..., :s] / (h * w), torch.ones(b, t, len(self.bg_centers), device=dev)), 2)
        vis = torch.cat((visible, torch.ones(b, t, len(self.bg_centers), dtype=torch.bool, device=dev)), 2)
        return boxes, frac, vis

    def forward(self, rgb, lab, frame_offset):
        """rgb uint8 [B,T,H,W,3]; lab uint8/long [B,T,H,W] -> tokens [B,N,width]."""
        b, t, h, w = lab.shape
        n, g = self.parents, self.glimpse
        lab = lab.long()
        with torch.autocast(device_type=lab.device.type, enabled=False):
            boxes, frac, vis = self.boxes(lab)
            u = torch.linspace(-1, 1, g, device=lab.device)
            gx = boxes[..., 0, None, None] + boxes[..., 2, None, None] * u[None, :]
            gy = boxes[..., 1, None, None] + boxes[..., 3, None, None] * u[:, None]
            grid = torch.stack(torch.broadcast_tensors(gx, gy), -1).reshape(b * t, n * g, g, 2)
            image = rgb.reshape(b * t, h, w, 3).permute(0, 3, 1, 2).float() / 255
            colors = F.grid_sample(image, grid, mode='nearest', align_corners=True)
            # Labels shifted by one so out-of-frame samples (zero padding) match no parent.
            labels = F.grid_sample((lab + 1).reshape(b * t, 1, h, w).float(), grid, mode='nearest',
                                   align_corners=True)
            own = torch.cat((torch.arange(self.slots, device=lab.device),
                             torch.full((n - self.slots,), self.slots, device=lab.device))) + 1
            mask = (labels.view(b * t, n, g, g) == own[None, :, None, None].float()).float()
            colors = colors.view(b * t, 3, n, g, g).permute(0, 2, 1, 3, 4)
            x = torch.cat((colors * mask[:, :, None], mask[:, :, None]), 2).reshape(b * t * n, 4, g, g)
            geo = torch.cat((fourier(boxes), torch.log1p(frac * 1000)[..., None] / math.log(1001),
                             vis.float()[..., None]), -1)
        tokens = self.cnn(x).view(b, t, n, -1) + self.geometry(geo)
        frames = torch.arange(frame_offset, frame_offset + t, device=lab.device)
        tokens = tokens + self.frame(frames)[None, :, None] + self.parent[:, None]
        seq = tokens.permute(0, 2, 1, 3).reshape(b * n, t, -1)
        out = self.temporal(seq).mean(1)
        return self.norm(out).view(b, n, -1)


class OracleDecoder(TrajectoryDecoder):
    def __init__(self, slots=SLOTS, bg_side=BG_SIDE, latent_dim=128, subkernels=4, texture_kernels=109):
        super().__init__(slots + bg_side ** 2, latent_dim, subkernels, texture_kernels, motion_init='static')
        self.slots = slots
        with torch.no_grad():
            self.anchors.copy_(torch.cat((torch.zeros(slots, 2), background_anchors(bg_side))))
        self.presence = nn.Linear(latent_dim, 1)
        nn.init.zeros_(self.presence.weight)
        nn.init.constant_(self.presence.bias, 3.)

    def decode(self, z):
        p = super().decode(z)
        logit = self.presence(z.float())[..., 0]
        p['log_presence'] = torch.cat((F.logsigmoid(logit[:, :self.slots]),
                                       torch.zeros_like(logit[:, self.slots:])), 1)
        return p

    def mixture(self, p, coords):
        """TrajectoryDecoder.render with presence; returns mixture weights [B,Q,N] and parent colors [B,Q,N,3]."""
        with torch.autocast(device_type=coords.device.type, enabled=False):
            coords = coords.float()
            local = self.local_coordinates(p, coords)
            delta = local[..., None, :] - p['child_center'][:, None]
            angle = p['child_angle'][:, None]
            c, s = angle.cos(), angle.sin()
            dx, dy = delta.unbind(-1)
            child = torch.stack((c * dx + s * dy, -s * dx + c * dy), -1)
            distance = (child / p['child_scale'][:, None]).square().sum(-1)
            log_density = -.5 * distance + p['child_log_amplitude'][:, None]
            parent_log_density = torch.logsumexp(log_density, dim=-1) + p['log_presence'][:, None]
            weights = parent_log_density.softmax(dim=-1)
            color = p['color'][:, None]
            d = (local[..., None, :] - p['texture_center'][:, None]) / p['texture_scale'][:, None]
            rbf = (-.5 * d.square().sum(-1)).exp()
            color = (color + (rbf[..., None] * p['texture_rgb'][:, None]).sum(-2)).sigmoid()
            return weights, color

    def render(self, p, coords):
        weights, color = self.mixture(p, coords)
        return (weights[..., None] * color).sum(-2)


class OracleCodec(nn.Module):
    def __init__(self, slots=SLOTS, bg_side=BG_SIDE, latent_dim=128, width=256, subkernels=4, texture_kernels=109):
        super().__init__()
        self.config = dict(variant='oracle_gt_masks', slots=slots, bg_side=bg_side, objects=slots + bg_side ** 2,
                           latent_dim=latent_dim, feature_width=width, subkernels=subkernels,
                           texture_kernels=texture_kernels, glimpse=GLIMPSE)
        self.slots, self.objects, self.latent_dim = slots, slots + bg_side ** 2, latent_dim
        self.encoder = GlimpseEncoder(slots, bg_side, width)
        self.mu = nn.Linear(width, latent_dim)
        self.logvar = nn.Linear(width, latent_dim)
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, -4.)
        self.decoder = OracleDecoder(slots, bg_side, latent_dim, subkernels, texture_kernels)

    def encode_future(self, rgb, lab, _unused=None):
        """Future frames 4..23 only: rgb uint8 [B,20,H,W,3], lab [B,20,H,W]."""
        h = self.encoder(rgb, lab, frame_offset=4)
        mu = F.layer_norm(self.mu(h).float(), (self.latent_dim,))
        logvar = self.logvar(h).float().clamp(-8, 2)
        z = mu + (.5 * logvar).exp() * torch.randn_like(mu)
        kl = .5 * (mu.square() + logvar.exp() - 1 - logvar).mean()
        return z, mu, logvar, kl

    def forward(self, rgb, lab, coords, target, labels):
        return codec_losses(self, rgb, lab, coords, target, labels)

    def render_queries(self, z, coords):
        with torch.autocast(device_type=z.device.type, enabled=False):
            return self.decoder.render(self.decoder.decode(z.float()), coords.float())

    def render_latent(self, z, size=64, frames=20, chunk=256, owners=False):
        tt, yy, xx = torch.meshgrid(torch.linspace(-1, 1, frames, device=z.device),
                                    torch.linspace(-1, 1, size, device=z.device),
                                    torch.linspace(-1, 1, size, device=z.device), indexing='ij')
        queries = torch.stack((xx, yy, tt), -1).reshape(1, -1, 3).expand(z.shape[0], -1, -1)
        rgb, own = [], []
        with torch.autocast(device_type=z.device.type, enabled=False):
            p = self.decoder.decode(z.float())
            for q in queries.split(chunk, dim=1):
                weights, color = self.decoder.mixture(p, q)
                rgb.append((weights[..., None] * color).sum(-2))
                if owners:
                    own.append(weights.argmax(-1))
        video = torch.cat(rgb, 1).reshape(z.shape[0], frames, size, size, 3).permute(0, 1, 4, 2, 3)
        if owners:
            return video, torch.cat(own, 1).reshape(z.shape[0], frames, size, size)
        return video


def query_batch(rgb, lab, points, generator, slots=SLOTS):
    """Half uniform future pixels, half uniform object pixels (uniform again if a clip has none).

    rgb [B,20,H,W,3] uint8, lab [B,20,H,W] -> coords [B,Q,3], rgb targets [B,Q,3], slot labels [B,Q].
    """
    b, t, h, w = lab.shape
    total = t * h * w
    flat_lab = lab.reshape(b, total).long()
    uniform = torch.randint(total, (b, points - points // 2), generator=generator).to(lab.device)
    obj = flat_lab < slots
    weights = torch.where(obj.any(1, keepdim=True), obj.float(), torch.ones_like(obj, dtype=torch.float))
    gen = torch.Generator(device=lab.device).manual_seed(int(torch.randint(2 ** 62, (1,), generator=generator)))
    focused = torch.multinomial(weights, points // 2, replacement=True, generator=gen)
    index = torch.cat((uniform, focused), 1)
    frames, rest = index // (h * w), index % (h * w)
    yy, xx = rest // w, rest % w
    coords = torch.stack((2 * xx.float() / (w - 1) - 1, 2 * yy.float() / (h - 1) - 1,
                          2 * frames.float() / (t - 1) - 1), -1)
    target = rgb.reshape(b, total, 3).gather(1, index[..., None].expand(-1, -1, 3)).float() / 255
    return coords, target, flat_lab.gather(1, index)


def codec_losses(codec, rgb_f, lab_f, coords, target, labels):
    z, mu, logvar, kl = codec.encode_future(rgb_f, lab_f)
    with torch.autocast(device_type=z.device.type, enabled=False):
        weights, color = codec.decoder.mixture(codec.decoder.decode(z.float()), coords)
        recon = F.mse_loss((weights[..., None] * color).sum(-2), target)
        s = codec.slots
        groups = torch.cat((weights[..., :s], weights[..., s:].sum(-1, keepdim=True)), -1)
        ownership = F.nll_loss(groups.clamp_min(1e-8).log().flatten(0, 1), labels.flatten())
        # GT-routed color: the labelled object's own parent, or the background parents' own mixture.
        bgw = weights[..., s:]
        bg_color = (bgw[..., None] * color[..., s:, :]).sum(-2) / bgw.sum(-1, keepdim=True).clamp_min(1e-8)
        obj_color = color[..., :s, :].gather(2, labels.clamp(max=s - 1)[..., None, None].expand(-1, -1, 1, 3))[..., 0, :]
        routed = torch.where((labels < s)[..., None], obj_color, bg_color)
        routed_mse = F.mse_loss(routed, target)
        accuracy = (groups.argmax(-1) == labels).float().mean()
    return {'recon': recon, 'routed': routed_mse, 'ownership': ownership, 'kl': kl, 'accuracy': accuracy}


class OracleObjectFlow(StableObjectFlow):
    """StableObjectFlow whose slot k also receives its own prefix token; context = all prefix tokens."""

    def __init__(self, objects, latent_dim, encoder_width, anchors, width=384, depth=8, heads=6,
                 slots=SLOTS, bg_side=BG_SIDE):
        super().__init__(objects, latent_dim, encoder_width, anchors, width, depth, heads)
        self.prefix = GlimpseEncoder(slots, bg_side, encoder_width)
        self.own = nn.Linear(encoder_width, width)

    def get_extra_state(self):
        return {**super().get_extra_state(), 'oracle': 'per-slot prefix glimpse tokens added to slot inputs'}

    def encode_prefix(self, rgb, lab):
        return self.prefix(rgb, lab, frame_offset=0)

    def forward(self, z, t, context):
        """context: prefix tokens [B,N,width], or (prefix rgb, prefix labels) to encode them here (training)."""
        with torch.autocast(device_type=z.device.type, dtype=torch.bfloat16, enabled=z.device.type == 'cuda'):
            if isinstance(context, tuple):
                context = self.encode_prefix(*context)
            half = self.width // 2
            f = torch.exp(-math.log(10000) * torch.arange(half, device=z.device).float() / max(half - 1, 1))
            phase = t.float()[:, None] * f[None] * 1000
            te = torch.cat([phase.sin(), phase.cos()], -1)
            te = self.time_mlp(te.to(z.dtype))
            x = self.input(z) + self.slots + self.own(context)
            ctx = self.condition(context)
            for block in self.blocks:
                x = block(x, ctx, te)
            return self.out(self.norm(x))


class OracleFlowModel(nn.Module):
    """The interface codec_flow_research.evaluate expects; sampling never calls the codec encoder."""

    def __init__(self, codec, flow):
        super().__init__()
        self.codec, self.flow = codec, flow
        self.codec.requires_grad_(False).eval()
        self.objects, self.latent_dim = codec.objects, codec.latent_dim

    def train(self, mode=True):
        super().train(mode)
        self.codec.eval()
        return self

    def prefix_context(self, rgb, lab):
        if rgb.shape[1] != 4 or lab.shape[1] != 4:
            raise ValueError('Prefix context takes exactly the 4 observed frames')
        return self.flow.encode_prefix(rgb, lab)

    @torch.no_grad()
    def target_latent(self, rgb, lab, _unused=None):
        return self.codec.encode_future(rgb, lab)

    def sample(self, context, initial_noise, steps):
        z = initial_noise
        for index in range(steps):
            t = torch.full((z.shape[0],), index / steps, device=z.device, dtype=torch.float32)
            z = z + self.flow(z, t, context).float() / steps
        return z
