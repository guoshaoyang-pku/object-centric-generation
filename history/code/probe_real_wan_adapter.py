"""Read-only, bounded real-Wan interface verification. No optimizer."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from wan_model import PretrainedJointObjectVideo
from wan_token_adapter import WanTokenAdapter
from trajectory_codec import TrajectoryDecoder

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--pretrained',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(20261027)
start=time.monotonic()
c=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
cfg=dict(c['args']);cfg['pretrained_path']=str(a.pretrained)
model=PretrainedJointObjectVideo(cfg)
missing=model.load_state_dict(c['model'],strict=False)
required={n for n,v in model.named_parameters() if v.requires_grad}
assert not missing.unexpected_keys and not required.intersection(missing.missing_keys)
model=model.cuda().eval()
z=torch.randn(1,250,128,device='cuda');context=torch.randn(1,128,256,device='cuda')
t=torch.tensor([.5],device='cuda')
legacy=model.flow
direct=WanTokenAdapter.from_legacy(legacy,position_mode='legacy_grid')
with torch.inference_mode():
    old=legacy(z,t,context).float()
    new=direct(z,t,context).float()
    delta=new-old
    relative=float(delta.square().mean().sqrt()/old.square().mean().sqrt().clamp_min(1e-8))
    forward={'max_abs':float(delta.abs().max()),'rms_relative':relative,'rms_original':float(old.square().mean().sqrt()),'finite':bool(torch.isfinite(new).all())}
    assert forward['finite'] and relative<.03,forward
anchors=TrajectoryDecoder(objects=250,latent_dim=128,subkernels=4,texture_kernels=0).anchors
objects=WanTokenAdapter.from_legacy(legacy,position_mode='object_xy',object_anchors=anchors)
out=objects(z,t,context)
loss=out.float().square().mean()
params=[objects.input.weight,objects.output.weight,objects.condition[0].weight]
grads=torch.autograd.grad(loss,params)
norms=[float(g.float().norm()) for g in grads]
assert all(torch.isfinite(g).all() for g in grads) and all(n>0 for n in norms)
result={'status':'complete','checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'step':c['step'],
        'legacy_forward_parity':forward,'admission_rms_relative_tolerance':.03,'object_xy_backward_gradient_norms':norms,
        'object_xy_extra_state':objects.get_extra_state(),'no_optimizer':True,'weights_updated':False,
        'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'elapsed_seconds':time.monotonic()-start,
        'peak_memory_bytes':torch.cuda.max_memory_allocated(),
        'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}}
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
