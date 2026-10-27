"""Frozen VAE target-seen reference, not a mathematical reconstruction bound."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from cache_vae_features import vae_fingerprint
from codec_research import CacheBank, atomic_json, digest, aggregate
from diagnose_persistence import time_metrics, save_gif


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['data','cache','pretrained','cohorts','out']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=300)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Use a new output directory')
    bank=CacheBank(a.data,a.cache)
    if vae_fingerprint(a.pretrained)!=bank.protocol['vae_fingerprint']:
        raise ValueError('VAE differs from cached encoder')
    co=json.loads(a.cohorts.read_text())
    if co['manifest_sha256']!=bank.manifest_sha:raise ValueError('Cohort binding differs')
    ids=[r['id'] for r in co['showcase4']]
    start=time.monotonic();a.out.mkdir(parents=True)
    torch.set_num_threads(4)
    vae=AutoencoderKLWan.from_pretrained(a.pretrained,subfolder='vae',torch_dtype=torch.float32,local_files_only=True).cuda().eval()
    vae.requires_grad_(False)
    mean=torch.tensor(vae.config.latents_mean,device='cuda').reshape(1,48,1,1,1)
    std=torch.tensor(vae.config.latents_std,device='cuda').reshape(1,48,1,1,1)
    rows=[];records=[]
    with torch.inference_mode():
        for identifier in ids:
            if time.monotonic()-start>a.max_seconds:raise TimeoutError('VAE reference budget')
            f=bank.features(identifier)['future']
            if f.shape!=(384,48):raise ValueError('Expected cached 6x8x8 future grid')
            z=torch.from_numpy(f).T.reshape(1,48,6,8,8).cuda()*std+mean
            decoded=vae.decode(z,return_dict=True).sample
            if decoded.shape[2]<20:raise ValueError('VAE returned too few frames')
            pred=((decoded[0,:,:20].permute(1,0,2,3).float()+1)/2).clamp(0,1).cpu().numpy()
            if pred.shape!=(20,3,128,128) or not np.isfinite(pred).all():raise ValueError('Bad decoded video')
            video=np.array(bank.rgb(identifier),copy=True).transpose(0,3,1,2).astype(np.float32)/255
            item={'id':identifier,'target_seen':True,'metrics':time_metrics(pred,video[4:],video[3])}
            records.append(item);atomic_json(a.out/(identifier+'.json'),item)
            rows.append((identifier,[video[4:].transpose(0,2,3,1),pred.transpose(0,2,3,1)]))
    save_gif(a.out/'comparison.gif',rows,['Ground truth','VAE FUTURE SEEN'],12)
    atomic_json(a.out/'summary.json',{'status':'complete','records':records,'means':aggregate(records),
        'data_manifest_sha256':bank.manifest_sha,'cache_protocol_sha256':bank.protocol_sha,
        'source_sha256':digest(Path(__file__)),'metric_size':128,'ids':ids,
        'elapsed_seconds':time.monotonic()-start,'no_optimizer':True,
        'scope':'Frozen VAE mode decode reference only. Not an information-theoretic lower bound; another decoder can reconstruct differently from the same features.'})
    atomic_json(a.out/'checksums.json',{f.name:digest(f) for f in a.out.iterdir() if f.is_file()})


if __name__=='__main__':main()
