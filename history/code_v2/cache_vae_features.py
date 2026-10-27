"""Cache frozen Wan VAE features only; no DiT, optimizer, or future-conditioned prefix.

Per-clip NPZ arrays are token-major: prefix/future [N,C], prefix_coords and
future_coords [N,3] in the original absolute x,y,t convention, and
future_local_coords [N,3] in the future window's local convention. Coordinates
and normalization reproduce PretrainedJointObjectVideo.visual_features.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def vae_fingerprint(pretrained):
    files=[]
    for p in sorted((pretrained/'vae').rglob('*')):
        if p.is_file() and (p.suffix in {'.safetensors','.bin'} or p.name.endswith('.json')):
            files.append({'path':str(p.relative_to(pretrained)),'bytes':p.stat().st_size,'sha256':digest(p)})
    if not files or not any(f['path'].endswith(('.safetensors','.bin')) for f in files):
        raise ValueError('Missing frozen VAE weight files')
    return {'files':files,'fingerprint_sha256':canonical_sha(files)}


def encode(vae,frames,mean,std):
    """frames [B,T,3,H,W] in [0,1]; matches original frozen FP32 mode encode."""
    import torch
    x=frames.permute(0,2,1,3,4).float()*2-1
    factor=int(getattr(vae.config,'scale_factor_temporal',4))
    pad=(factor-(x.shape[2]-1)%factor)%factor
    if pad:x=torch.cat([x,x[:,:,-1:].expand(-1,-1,pad,-1,-1)],dim=2)
    with torch.autocast(device_type=x.device.type,enabled=False):
        latent=vae.encode(x,return_dict=True).latent_dist.mode()
        return (latent-mean)/std.clamp_min(1e-6)


def coordinates(latent_shape,frames,time_start,context_frames=4,future_frames=20):
    import torch
    _,_,t,h,w=latent_shape
    zz,yy,xx=torch.meshgrid(torch.linspace(0,max(frames-1,1),t),
        torch.linspace(-1,1,h),torch.linspace(-1,1,w),indexing='ij')
    absolute=torch.stack([xx,yy,2*(zz+time_start)/max(context_frames+future_frames-1,1)-1],-1).reshape(-1,3)
    local=torch.stack([xx,yy,2*zz/max(frames-1,1)-1],-1).reshape(-1,3)
    return absolute.numpy(),local.numpy()


def valid_existing(out,record,protocol_sha):
    import numpy as np
    meta_path=out/'records'/record['split']/(record['id']+'.json')
    npz_path=out/'features'/record['split']/(record['id']+'.npz')
    if not meta_path.exists():return None
    item=json.loads(meta_path.read_text())
    if item['protocol_sha256']!=protocol_sha or item['source_array_sha256']!=record['array_sha256']:
        raise ValueError('Existing cache provenance differs: '+record['id'])
    if not npz_path.exists() or digest(npz_path)!=item['features_sha256']:
        raise ValueError('Existing feature cache checksum mismatch: '+record['id'])
    with np.load(npz_path,allow_pickle=False) as arrays:
        if set(arrays.files)!=set(item['arrays']):raise ValueError('Unexpected NPZ fields')
        for key,spec in item['arrays'].items():
            value=arrays[key]
            if list(value.shape)!=spec['shape'] or str(value.dtype)!=spec['dtype'] or not np.isfinite(value).all():
                raise ValueError('Invalid cached array '+record['id']+'/'+key)
    return item


def run(a):
    import numpy as np
    import torch
    started=time.time();cutoff=min(started+a.max_hours*3600,a.deadline_unix or float('inf'))
    if cutoff<=started:raise TimeoutError('Cache deadline already passed')
    a.out.mkdir(parents=True,exist_ok=True)
    lock=(a.out/f'.shard_{a.shard:03d}_of_{a.shards:03d}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=a.data/'manifest.json';manifest=json.loads(source.read_text())
    if manifest.get('test_split_used'):raise ValueError('Only official train/validation are admitted')
    all_records=sorted(manifest['samples'],key=lambda r:(r['split'],r['id']))
    if len({r['id'] for r in all_records})!=len(all_records):raise ValueError('Duplicate clip IDs')
    admitted=[]
    for split,limit in [('train',a.limit_train),('validation',a.limit_val)]:
        subset=[r for r in all_records if r['split']==split]
        if limit is not None:subset=subset[:limit]
        admitted.extend(subset)
    admitted_ids={r['id'] for r in admitted}
    selected=[r for i,r in enumerate(all_records) if r['id'] in admitted_ids and i%a.shards==a.shard]
    for r in selected:
        if Path(r['id']).name!=r['id'] or Path(r['array']).parts[0]!=r['split'] or '..' in Path(r['array']).parts:
            raise ValueError('Unsafe source record path')
        if not r.get('official_split') or r['shape']!=[24,128,128,3] or r.get('dtype')!='uint8':
            raise ValueError('Source contract requires official uint8 24x128x128 RGB')
    fingerprint=vae_fingerprint(a.pretrained)
    protocol={'version':'wan-vae-feature-cache-v1','source_manifest_sha256':digest(source),
        'source_script_sha256':digest(Path(__file__)),'vae_fingerprint':fingerprint,
        'context_frames':4,'future_frames':20,'dtype':a.dtype,'encode_dtype':'float32',
        'vae_sampling':'posterior mode; frozen eval; no_grad',
        'padding':'repeat last observed frame separately within each prefix/future window to temporal_stride*k+1',
        'normalization':'(posterior.mode - config.latents_mean) / clamp_min(config.latents_std,1e-6)',
        'coordinates':'x,y linspace[-1,1]; z linspace[0,frames-1] over actual latent temporal length; absolute 2*(z+start)/23-1; future local 2*z/19-1',
        'ordering':'latent.flatten(2).transpose(1,2); coordinates meshgrid(t,y,x)',
        'shard_partition':'full manifest sorted (split,id), original index modulo --shards; optional per-split limits only filter, never renumber',
        'model_kind':'AutoencoderKLWan only; no transformer loaded'}
    # Different shards synchronize only this short protocol check/create section.
    with (a.out/'.protocol.lock').open('a') as shared:
        fcntl.flock(shared,fcntl.LOCK_EX)
        path=a.out/'protocol.json'
        if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Output cache protocol differs')
        if not path.exists():atomic_json(path,protocol)
    protocol_sha=canonical_sha(protocol)
    state_path=a.out/f'shard_{a.shard:03d}_of_{a.shards:03d}.json'
    items=[];pending=[];vae=None;encoded=0
    def publish(status,error=None):
        result={'version':protocol['version'],'protocol_sha256':protocol_sha,'status':status,'shard':a.shard,'shards':a.shards,
            'selected_count':len(selected),'complete_count':len(items),'encoded_this_run':encoded,
            'source_data':str(a.data.resolve()),'pretrained':str(a.pretrained.resolve()),'dtype':a.dtype,
            'torch_version':torch.__version__,'limits':{'train':a.limit_train,'validation':a.limit_val},
            'started_unix':started,'updated_unix':time.time(),'deadline_unix':cutoff,
            'elapsed_seconds':time.time()-started,'allocated_gpu_hours':(time.time()-started)/3600 if vae is not None else 0,
            'gpu_hours_definition':'one selected GPU elapsed wall time from process start; includes load/hash/cache I/O; may overlap other tasks',
            'records':items}
        if error is not None:result['error']=error
        atomic_json(state_path,result)
    try:
        for record in selected:
            if time.time()>=cutoff:publish('partial_deadline');return
            # Always verify the source too, including on resumed cache reads.
            if digest(a.data/record['array'])!=record['array_sha256']:raise ValueError('Source array SHA mismatch: '+record['id'])
            item=valid_existing(a.out,record,protocol_sha)
            if item is None:pending.append(record)
            else:items.append(item)
        publish('verified_existing')
        if not pending:publish('complete');return
        if time.time()>=cutoff:publish('partial_deadline');return
        from diffusers import AutoencoderKLWan
        torch.set_num_threads(4);torch.cuda.set_device(0)
        vae=AutoencoderKLWan.from_pretrained(a.pretrained,subfolder='vae',torch_dtype=torch.float32,local_files_only=True).cuda().float().eval()
        vae.requires_grad_(False)
        channels=int(vae.config.z_dim)
        if channels!=48:raise ValueError('Expected Wan2.2 TI2V 48-channel VAE')
        mean=torch.tensor(vae.config.latents_mean,device='cuda',dtype=torch.float32).reshape(1,channels,1,1,1)
        std=torch.tensor(vae.config.latents_std,device='cuda',dtype=torch.float32).reshape(1,channels,1,1,1)
        for offset in range(0,len(pending),a.batch):
            if time.time()>=cutoff:publish('partial_deadline');return
            batch=pending[offset:offset+a.batch];values=[]
            for r in batch:
                v=np.load(a.data/r['array'],allow_pickle=False)
                if v.shape!=(24,128,128,3) or v.dtype!=np.uint8:raise ValueError('Unexpected source RGB shape/dtype')
                values.append(v)
            video=torch.from_numpy(np.stack(values)).permute(0,1,4,2,3).cuda().float()/255
            with torch.inference_mode():
                # Prefix encoder sees only prefix. Future has a separate VAE call.
                prefix=encode(vae,video[:,:4],mean,std)
                future=encode(vae,video[:,4:24],mean,std)
            pc,_=coordinates(prefix.shape,4,0);fc,flc=coordinates(future.shape,20,4)
            dtype=np.float32 if a.dtype=='float32' else np.float16
            p=prefix.flatten(2).transpose(1,2).cpu().numpy().astype(dtype)
            f=future.flatten(2).transpose(1,2).cpu().numpy().astype(dtype)
            if not np.isfinite(p).all() or not np.isfinite(f).all():raise ValueError('Nonfinite cached feature values')
            for i,r in enumerate(batch):
                relative=Path('features')/r['split']/(r['id']+'.npz');path=a.out/relative;path.parent.mkdir(parents=True,exist_ok=True)
                arrays={'prefix':p[i],'future':f[i],'prefix_coords':pc,'future_coords':fc,'future_local_coords':flc}
                temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
                with temporary.open('wb') as stream:np.savez(stream,**arrays)
                os.replace(temporary,path)
                item={'id':r['id'],'split':r['split'],'source_array':r['array'],'source_array_sha256':r['array_sha256'],
                    'source_url':r['source_url'],'features':str(relative),'features_sha256':digest(path),
                    'protocol_sha256':protocol_sha,'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in arrays.items()}}
                atomic_json(a.out/'records'/r['split']/(r['id']+'.json'),item);items.append(item);encoded+=1
            publish('running')
            print(json.dumps({'shard':a.shard,'completed':len(items),'selected':len(selected),'encoded_this_run':encoded,
                'prefix_shape':list(p.shape[1:]),'future_shape':list(f.shape[1:]),'elapsed_seconds':time.time()-started}),flush=True)
            del video,prefix,future,p,f
        items.sort(key=lambda r:(r['split'],r['id']));publish('complete')
    except Exception as exc:
        publish('failed',type(exc).__name__+': '+str(exc));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','pretrained','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--shards',type=int,default=1);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--max-hours',type=float,default=1)
    p.add_argument('--deadline-unix',type=float);p.add_argument('--dtype',choices=['float32','float16'],default='float32')
    p.add_argument('--limit-train',type=int);p.add_argument('--limit-val',type=int)
    a=p.parse_args()
    if a.shards<1 or not 0<=a.shard<a.shards or a.batch<1 or a.max_hours<=0:p.error('Invalid shard, batch or time bounds')
    if any(x is not None and x<0 for x in [a.limit_train,a.limit_val]):p.error('Limits must be nonnegative')
    run(a)


if __name__=='__main__':main()
