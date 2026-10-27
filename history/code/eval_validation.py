"""Preregistered full-validation continuation evaluation; read-only inference.

Two fixed master seeds are averaged WITHIN each clip before clip bootstrap.
All predictions receive prefix only. Future-AE reconstruction is an explicitly
labelled representation diagnostic. No FVD, physical or semantic claims.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.nn import functional as F

EVAL_PROTOCOL_VERSION='movi-e-continuation-validation-v1.1'


def digest(file):
    h=hashlib.sha256()
    with open(file,'rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def atomic_json(file,value):
    file=Path(file);file.parent.mkdir(parents=True,exist_ok=True);tmp=file.with_name(file.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n');os.replace(tmp,file)


def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def clip_seed(master,identifier):
    b=hashlib.sha256(f'object-continuation-eval-v1|{master}|{identifier}'.encode()).digest()
    return int.from_bytes(b[:8],'big')%(2**63-1)


def clip_noise(ids,master,objects,latent_dim,device,dtype):
    # CPU FP32 generators make the noise definition independent of batch size,
    # GPU model and the final short batch. Only the final dtype cast is applied.
    values=[]
    for identifier in ids:
        g=torch.Generator(device='cpu').manual_seed(clip_seed(master,identifier))
        values.append(torch.randn(objects,latent_dim,generator=g,dtype=torch.float32))
    return torch.stack(values).to(device=device,dtype=dtype)


def cohort(data,expected):
    manifest=json.loads((data/'manifest.json').read_text())
    if manifest.get('test_split_used'):
        raise ValueError('This protocol only uses official validation, never test')
    records=[]
    for s in manifest['samples']:
        if s['split']!='validation':continue
        if not s.get('official_split',False) or 'movi_e-validation' not in s['source_url']:
            raise ValueError('Validation provenance is not official MOVi-E')
        records.append({'id':s['id'],'array':s['array'],'sha256':s['array_sha256'],'shape':s['shape'],
                        'dtype':s.get('dtype','uint8'),'split':'validation','source_url':s['source_url']})
    records.sort(key=lambda x:x['id'])
    if len(records)!=expected or len({r['id'] for r in records})!=expected:
        raise ValueError(f'Expected {expected} unique validation clips, got {len(records)}')
    if len({r['array'] for r in records})!=expected or len({r['sha256'] for r in records})!=expected:
        raise ValueError('Validation cohort contains duplicate array paths or identical clip contents')
    return records,manifest


def check_array(data,r):
    f=data/r['array']
    if not r['array'].startswith('validation/') or digest(f)!=r['sha256']:
        raise ValueError(f'Unverified validation array: {f}')
    a=np.load(f,allow_pickle=False)
    if list(a.shape)!=r['shape'] or a.shape!=(24,128,128,3) or str(a.dtype)!=r['dtype'] or a.dtype!=np.uint8:
        raise ValueError(f'Invalid RGB array shape/dtype: {f}')
    return a


def prepare(a):
    if a.checkpoint_role=='final':
        expected_step=20000 if a.expected_step is None else a.expected_step
        if expected_step!=20000:raise ValueError('Final protocol is fixed to step20000; use a separate candidate protocol')
    else:
        expected_step=a.expected_step
        if expected_step is not None and not 0<expected_step<=20000:raise ValueError('Candidate step must be in1..20000')
    records,manifest=cohort(a.data,a.expected_clips)
    for r in records:check_array(a.data,r)
    selected=np.linspace(0,len(records)-1,min(4,len(records)),dtype=int).tolist()
    p={'protocol':EVAL_PROTOCOL_VERSION,'created_unix':time.time(),
       'revision_note':'2026-10-26 integrity hardening and explicit final/candidate binding; cohort, seeds and metric definitions unchanged',
       'scope':'Official validation diagnostics; validation has also been used for training previews. Not a blind test or physical-fidelity benchmark.',
       'source_data':str(a.data.resolve()),'source_manifest_sha256':digest(a.data/'manifest.json'),
       'validation_fingerprint_sha256':canonical_sha(records),'validation_clips':records,
       'context_frames':4,'future_frames':20,'input_size':128,'metric_size':a.size,'fps':12,
       'objects':250,'subkernels':4,'latent_dim':128,'texture_kernels':64,'encoder_width':256,
       'posterior_locality':True,'posterior_sigma':[.4,.5,.5],'lora_rank':8,'lora_alpha':8.,'sampling_steps':a.steps,
       'master_seeds':a.seeds,'seed_algorithm':'SHA256(object-continuation-eval-v1|master|clipID), first8big-endian bytes modulo(2^63-1); CPUtorch.randn FP32 then cast',
       'clip_seed_map':{r['id']:{str(s):str(clip_seed(s,r['id'])) for s in a.seeds} for r in records},
       'representative_indices':selected,'representative_ids':[records[i]['id'] for i in selected],
       'representative_rule':'four evenly spaced indices in sorted official validation IDs, selected before evaluation outputs',
       'preview_size':a.preview_size,'bootstrap_resamples':a.bootstrap,'bootstrap_seed':20261026,
       'bootstrap_unit':'clip; two seeds averaged within each clip; no best-of-two/oracle selection',
       'checkpoint_role':a.checkpoint_role,'expected_checkpoint_step':expected_step,
       'candidate_selection_rule':a.candidate_selection_rule if a.checkpoint_role=='candidate' else None,
       'candidate_binding':'explicit protocol step, or an external selection JSON binding step+checkpoint SHA before full250 inference' if a.checkpoint_role=='candidate' else None,
       'prepare_source_sha256':digest(Path(__file__))}
    target=a.out/'protocol.json'
    if target.exists():
        old=json.loads(target.read_text())
        if old['protocol']!=EVAL_PROTOCOL_VERSION or old['prepare_source_sha256']!=p['prepare_source_sha256']:
            raise ValueError('Protocol/source revision differs; preserve the old file and use a new explicit protocol directory')
        for key in ['validation_fingerprint_sha256','context_frames','future_frames','metric_size','sampling_steps','master_seeds','representative_ids','expected_checkpoint_step','bootstrap_resamples']:
            if old[key]!=p[key]:raise ValueError(f'Refusing to overwrite a different preregistered protocol: {key}')
        if old.get('checkpoint_role','final')!=p['checkpoint_role']:raise ValueError('Checkpoint role differs from existing protocol')
        if old.get('candidate_selection_rule')!=p.get('candidate_selection_rule'):raise ValueError('Candidate selection rule changed')
        print(json.dumps({'status':'already_preregistered','path':str(target),'protocol_sha256':digest(target)}));return
    atomic_json(target,p);print(json.dumps({'status':'preregistered','path':str(target),'protocol_sha256':digest(target),'clips':len(records),'representatives':p['representative_ids']}),flush=True)


def describe(a):
    return {'spatial_contrast_std':float(a.std(axis=(-2,-1)).mean()),
            'temporal_change_l1':float(np.abs(np.diff(a,axis=0)).mean()),
            'edge_strength_l1':float((np.abs(np.diff(a,axis=-1)).mean()+np.abs(np.diff(a,axis=-2)).mean())/2)}


def cos(a,b):
    x=a.reshape(-1).astype(np.float64);y=b.reshape(-1).astype(np.float64);den=np.linalg.norm(x)*np.linalg.norm(y)
    return float(np.dot(x,y)/den) if den>1e-12 else None


def metrics(pred,truth):
    mse=float(np.mean((pred.astype(np.float64)-truth.astype(np.float64))**2))
    return {'mse':mse,'psnr':-10*math.log10(max(mse,1e-12)),**describe(pred),
            'pixel_correlation':cos(pred-pred.mean(),truth-truth.mean()),
            'temporal_delta_cosine':cos(np.diff(pred,axis=0),np.diff(truth,axis=0))}


def average_seed_metrics(values):
    out={}
    for key in values[0]:
        valid=[v[key] for v in values if v[key] is not None]
        out[key]=float(np.mean(valid)) if valid else None
    # Primary PSNR refers to the within-clip mean MSE, not the best seed.
    out['psnr']=-10*math.log10(max(out['mse'],1e-12))
    return out


def interval(values,iterations,seed):
    clean=np.array([x for x in values if x is not None and math.isfinite(x)],np.float64)
    if not len(clean):return {'mean':None,'ci95':None,'valid_clips':0}
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(clean),(iterations,len(clean)))
    means=clean[indices].mean(1);return {'mean':float(clean.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'valid_clips':len(clean)}


def summarize(records,protocol):
    names=['persistence','future_AE']+[f'sample_seed{s}' for s in protocol['master_seeds']]+['sample_seed_average']
    result={'clips':len(records),'bootstrap_unit':'clip, with two-seed metrics averaged within clip before resampling',
            'bootstrap_resamples':protocol['bootstrap_resamples'],'bootstrap_seed':protocol['bootstrap_seed'],'methods':{},'paired_mse_difference_vs_persistence':{}}
    for name in names:
        keys=records[0]['methods'][name]
        summary={k:interval([r['methods'][name][k] for r in records],protocol['bootstrap_resamples'],protocol['bootstrap_seed']) for k in keys}
        mse=summary['mse'];lo,hi=mse['ci95']
        summary['psnr_from_mean_mse']={'mean':-10*math.log10(max(mse['mean'],1e-12)),
                                     'ci95':[-10*math.log10(max(hi,1e-12)),-10*math.log10(max(lo,1e-12))],'valid_clips':len(records)}
        result['methods'][name]=summary
        if name!='persistence':
            delta=[r['methods'][name]['mse']-r['methods']['persistence']['mse'] for r in records]
            result['paired_mse_difference_vs_persistence'][name]=interval(delta,protocol['bootstrap_resamples'],protocol['bootstrap_seed'])
    result['interpretation']='MSE/contrast/edge/motion-change/correlation describe rendered image agreement. Temporal cosine is undefined for zero-motion sequences and reports valid N. These metrics do not establish physical validity or distributional video quality.'
    return result


def frozen_fingerprint(path):
    files=[]
    for sub in ('transformer','vae'):
        for f in sorted((path/sub).rglob('*')):
            if f.is_file() and (f.suffix in {'.safetensors','.bin'} or f.name.endswith('.json')):
                files.append({'path':str(f.relative_to(path)),'bytes':f.stat().st_size,'sha256':digest(f)})
    return {'files':files,'fingerprint_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}


def representative_gif(prefix,truth,ae,persistence,samples,identifier,protocol,file):
    # One preregistered clip per GIF, five labelled panels. Never save all250 GIFs.
    size=128;header=30;rows=2;cols=3;frames=[]
    labels=['Ground truth','Persistence','Future AE (target seen)']+[f'Prefix sample seed{s}' for s in protocol['master_seeds']]
    videos=[truth,persistence,ae]+[samples[s] for s in protocol['master_seeds']]
    p=protocol['preview_size']
    def small(a):
        t=torch.from_numpy(a).float();return F.interpolate(t,(p,p),mode='bilinear',align_corners=False).numpy()
    prefix=small(prefix);videos=[np.concatenate([prefix,small(v)],0) for v in videos]
    for i in range(videos[0].shape[0]):
        im=Image.new('RGB',(cols*size,rows*(size+header)),(20,20,20));d=ImageDraw.Draw(im)
        for j,v in enumerate(videos):
            x=(j%cols)*size;y=(j//cols)*(size+header)
            label=labels[j]
            d.text((x+3,y+2),label[:23],fill='white');d.text((x+3,y+14),f'{"context" if i<4 else "future"} {i/12:.2f}s',fill='white')
            a=(v[i].transpose(1,2,0).clip(0,1)*255).round().astype('uint8')
            im.paste(Image.fromarray(a).resize((size,size),Image.Resampling.NEAREST),(x,y+header))
        x=2*size;y=size+header;d.text((x+4,y+8),identifier[:22],fill='white');d.text((x+4,y+28),f'metrics {protocol["metric_size"]}px',fill='white');d.text((x+4,y+46),f'preview {p}px; {protocol["sampling_steps"]} steps',fill='white')
        frames.append(im)
    duration=[round((i+1)*100/12)*10-round(i*100/12)*10 for i in range(len(frames))]
    frames[0].save(file,save_all=True,append_images=frames[1:],duration=duration,loop=0,optimize=False,disposal=2)
    frames[len(frames)//2].save(file.with_suffix('.png'))


def read_completed(path):
    if not path.exists():return {}
    raw=path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        # A process interruption may leave only its last owned JSONL line
        # incomplete. Drop that line and deterministically recompute the clip.
        raw=raw[:raw.rfind(b'\n')+1]
        path.write_bytes(raw)
    out={}
    for line in raw.decode().splitlines():
        if not line:continue
        r=json.loads(line)
        if r['id'] in out:raise ValueError('Duplicate per-clip evaluation record')
        out[r['id']]=r
    return out


def validate_completed(done,records,protocol,identity):
    registered={r['id']:r for r in records}
    expected_methods={'persistence','future_AE','sample_seed_average'}|{f'sample_seed{s}' for s in protocol['master_seeds']}
    metric_keys={'mse','psnr','spatial_contrast_std','temporal_change_l1','edge_strength_l1','pixel_correlation','temporal_delta_cosine'}
    for identifier,item in done.items():
        if identifier not in registered:raise ValueError(f'Unregistered completed clip: {identifier}')
        if item.get('array_sha256')!=registered[identifier]['sha256'] or item.get('seeds')!=protocol['clip_seed_map'][identifier]:
            raise ValueError(f'Completed clip data hash or seed map differs: {identifier}')
        if any(item.get(k)!=identity[k] for k in identity):
            raise ValueError(f'Completed clip belongs to a different checkpoint/protocol: {identifier}')
        if set(item.get('methods',{}))!=expected_methods:raise ValueError(f'Incomplete method records: {identifier}')
        if set(item.get('ground_truth',{}))!={'spatial_contrast_std','temporal_change_l1','edge_strength_l1'}:
            raise ValueError('Ground-truth descriptor schema differs')
        if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in item['ground_truth'].values()):
            raise ValueError('Invalid ground-truth descriptors')
        for method,value in item['methods'].items():
            if set(value)!=metric_keys:raise ValueError(f'Metric schema differs: {identifier}/{method}')
            for key,number in value.items():
                if number is None and key in {'pixel_correlation','temporal_delta_cosine'}:continue
                if not isinstance(number,(float,int)) or not math.isfinite(number):raise ValueError('Invalid metric value')
                if key in {'mse','spatial_contrast_std','temporal_change_l1','edge_strength_l1'} and number<0:raise ValueError('Negative error/strength metric')
            expected_psnr=-10*math.log10(max(value['mse'],1e-12))
            if not math.isclose(value['psnr'],expected_psnr,rel_tol=1e-9,abs_tol=1e-9):raise ValueError('PSNR does not match stored MSE')
        averaged=average_seed_metrics([item['methods'][f'sample_seed{s}'] for s in protocol['master_seeds']])
        for key,value in averaged.items():
            stored=item['methods']['sample_seed_average'][key]
            if value is None:
                if stored is not None:raise ValueError('Undefined seed-average metric changed')
            elif stored is None or not math.isclose(value,stored,rel_tol=1e-9,abs_tol=1e-12):
                raise ValueError('Within-clip seed average differs from its two recorded seeds')


@torch.no_grad()
def evaluate(a):
    start=time.time();protocol=json.loads(a.protocol.read_text());protocol_sha=digest(a.protocol)
    if protocol['protocol']!=EVAL_PROTOCOL_VERSION or protocol['prepare_source_sha256']!=digest(Path(__file__)):
        raise ValueError('Evaluation script differs from the explicitly frozen protocol source')
    records,_=cohort(a.data,len(protocol['validation_clips']))
    if canonical_sha(records)!=protocol['validation_fingerprint_sha256']:raise ValueError('Validation cohort changed since preregistration')
    if digest(a.checkpoint)!=a.checkpoint_sha:raise ValueError('Explicit checkpoint SHA mismatch')
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    role=protocol.get('checkpoint_role','final');selection=None;expected=protocol['expected_checkpoint_step']
    if role=='final':
        if expected!=20000 or a.candidate_selection_file:raise ValueError('Final protocol remains strictly bound to step20000')
    elif role=='candidate':
        if a.candidate_selection_file:
            selection=json.loads(a.candidate_selection_file.read_text())
            selected_step=selection.get('checkpoint_step',selection.get('step'))
            selected_sha=selection.get('checkpoint_sha256',selection.get('sha256'))
            if not isinstance(selected_step,int) or not 0<selected_step<=20000 or selected_sha!=a.checkpoint_sha:
                raise ValueError('Candidate selection file does not bind this step+SHA')
            if not selection.get('selection_rule'):raise ValueError('Candidate descriptor must state its proxy selection rule')
            if expected is not None and expected!=selected_step:raise ValueError('Candidate descriptor differs from explicit protocol step')
            expected=selected_step
        if expected is None:raise ValueError('Candidate evaluation requires an explicit bound step or selection descriptor file')
    else:raise ValueError('Unknown checkpoint role')
    if checkpoint['step']!=expected:raise ValueError('Checkpoint step differs from the explicit final/candidate protocol binding')
    cfg=dict(checkpoint['args']);cfg['pretrained_path']=str(a.pretrained)
    for key in ('context_frames','future_frames','objects','subkernels','latent_dim','texture_kernels','encoder_width','posterior_locality','lora_rank','lora_alpha'):
        if cfg[key]!=protocol[key]:raise ValueError(f'Architecture differs from protocol: {key}')
    if list(cfg['posterior_sigma'])!=protocol['posterior_sigma']:raise ValueError('Posterior locality scales changed')
    frozen=frozen_fingerprint(a.pretrained)
    if frozen['fingerprint_sha256']!=cfg['pretrained_fingerprint_sha256']:raise ValueError('Frozen backbone/VAE differs from training checkpoint')
    # Rehash/load all250 source arrays before accepting any resumed clip record.
    arrays_by_id={r['id']:check_array(a.data,r) for r in records}
    a.out.mkdir(parents=True,exist_ok=True);identity={'checkpoint_sha256':a.checkpoint_sha,'protocol_sha256':protocol_sha}
    if selection is not None:identity['candidate_selection_sha256']=digest(a.candidate_selection_file)
    if (a.out/'identity.json').exists() and json.loads((a.out/'identity.json').read_text())!=identity:raise ValueError('Refusing mixed checkpoint/protocol outputs')
    atomic_json(a.out/'identity.json',identity)
    done=read_completed(a.out/'per_clip.jsonl');validate_completed(done,records,protocol,identity)
    pending=[r for r in records if r['id'] not in done]
    if not pending:
        summary=summarize([done[r['id']] for r in records],protocol)
        summary.update(**identity,checkpoint_role=role,checkpoint_step=checkpoint['step'],completed_from_existing_records=True)
        atomic_json(a.out/'summary.json',summary)
        atomic_json(a.out/'progress.json',{'status':'complete','completed_clips':len(done),'total_clips':len(records),'finished':time.time(),'completed_from_existing_records':True})
        return
    from wan_model import PretrainedJointObjectVideo
    torch.set_num_threads(4);torch.cuda.set_device(0);torch.manual_seed(20261026)
    model=PretrainedJointObjectVideo(cfg)
    incompatible=model.load_state_dict(checkpoint['model'],strict=False)
    trainable={n for n,v in model.named_parameters() if v.requires_grad}
    persistent=set(model.state_dict())
    required_buffers={n for n,_ in model.named_buffers() if n in persistent and not n.startswith(('vae.','flow.backbone.'))}
    if (trainable|required_buffers).intersection(incompatible.missing_keys) or incompatible.unexpected_keys:raise ValueError('Checkpoint trainable/buffer state mismatch')
    model=model.cuda().eval()
    provenance={'checkpoint':str(a.checkpoint),'checkpoint_step':checkpoint['step'],**identity,'checkpoint_args':checkpoint['args'],
                'evaluation_data':str(a.data.resolve()),'evaluation_manifest_sha256':digest(a.data/'manifest.json'),
                'protocol':protocol,'frozen_pretrained':frozen,'source_sha256':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
                'model_pretrained_info':model.pretrained_info,'parameter_counts':model.parameter_counts(),
                'torch':torch.__version__,'device':torch.cuda.get_device_name(0),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                'checkpoint_role':role,'candidate_selection':selection,'evaluation_batch_size':a.batch_size,
                'sampling':'Euler FM, 250 object latents, prefix only, CPU per-clip FP32 noise then cast, no CFG',
                'role':'evaluation only; no optimizer or model updates','started':time.time()}
    atomic_json(a.out/'provenance.json',provenance);atomic_json(a.out/'protocol.json',protocol)
    reps=set(protocol['representative_ids']);repdir=a.out/'representatives';repdir.mkdir(exist_ok=True)
    def resize(v):return F.interpolate(v.flatten(0,1),(protocol['metric_size'],protocol['metric_size']),mode='bilinear',align_corners=False).reshape(v.shape[0],v.shape[1],3,protocol['metric_size'],protocol['metric_size'])
    for offset in range(0,len(pending),a.batch_size):
        if time.time()-start>a.max_seconds:raise TimeoutError('Evaluation time budget reached; per-clip records can resume')
        batch=pending[offset:offset+a.batch_size];ids=[r['id'] for r in batch]
        arrays=[arrays_by_id[r['id']] for r in batch]
        full=torch.from_numpy(np.stack(arrays)).permute(0,1,4,2,3).cuda().float()/255
        context=full[:,:4];future=full[:,4:24]
        truth=resize(future).float().cpu().numpy();prefix=resize(context).float().cpu().numpy()
        persistence=np.repeat(prefix[:,-1:],20,axis=1)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            condition=model.visual_features(context,time_start=0)
            _,mu,_,_=model._future_posterior(future)
            reconstruction=model.render_latent(mu,protocol['metric_size'],protocol['metric_size']).float().cpu().numpy()
            generated={}
            for seed in protocol['master_seeds']:
                z=clip_noise(ids,seed,model.objects,model.latent_dim,condition.device,condition.dtype)
                for step in range(protocol['sampling_steps']):
                    t=torch.full((len(batch),),step/protocol['sampling_steps'],device=z.device,dtype=z.dtype)
                    z=z+model.flow(z,t,condition)/protocol['sampling_steps']
                generated[seed]=model.render_latent(z,protocol['metric_size'],protocol['metric_size']).float().cpu().numpy()
        if not all(np.isfinite(v).all() for v in [reconstruction,*generated.values()]):raise ValueError('Nonfinite decoded output')
        batch_records=[]
        for i,r in enumerate(batch):
            methods={'persistence':metrics(persistence[i],truth[i]),'future_AE':metrics(reconstruction[i],truth[i])}
            methods.update({f'sample_seed{s}':metrics(generated[s][i],truth[i]) for s in protocol['master_seeds']})
            methods['sample_seed_average']=average_seed_metrics([methods[f'sample_seed{s}'] for s in protocol['master_seeds']])
            item={'id':r['id'],'array_sha256':r['sha256'],'seeds':protocol['clip_seed_map'][r['id']],
                  **identity,'actual_batch_size':len(batch),
                  'ground_truth':describe(truth[i]),'methods':methods}
            if r['id'] in reps:
                representative_gif(prefix[i],truth[i],reconstruction[i],persistence[i],{s:generated[s][i] for s in generated},r['id'],protocol,repdir/(r['id']+'.gif'))
            batch_records.append(item)
        with (a.out/'per_clip.jsonl').open('a') as f:
            for item in batch_records:f.write(json.dumps(item,allow_nan=False)+'\n');done[item['id']]=item
            f.flush();os.fsync(f.fileno())
        progress={'status':'running','completed_clips':len(done),'total_clips':len(records),'elapsed_seconds':time.time()-start,
                  'last_ids':ids,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}
        atomic_json(a.out/'progress.json',progress);print(json.dumps(progress),flush=True)
    ordered=[done[r['id']] for r in records];summary=summarize(ordered,protocol)
    summary.update(**identity,checkpoint_role=role,checkpoint_step=checkpoint['step'],last_attempt_seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
    atomic_json(a.out/'summary.json',summary);atomic_json(a.out/'progress.json',{'status':'complete','completed_clips':len(done),'total_clips':len(records),'finished':time.time()})
    print(json.dumps({'status':'complete','clips':len(done),'seconds':time.time()-start}),flush=True)


def self_check(a):
    ids=['validation_a','validation_b','validation_c','validation_d']
    full=clip_noise(ids,17,7,9,'cpu',torch.float32)
    parts=torch.cat([clip_noise(ids[:3],17,7,9,'cpu',torch.float32),clip_noise(ids[3:],17,7,9,'cpu',torch.float32)])
    assert torch.equal(full,parts)
    assert not torch.equal(full,clip_noise(ids,29,7,9,'cpu',torch.float32))
    x=np.random.default_rng(2).random((3,3,8,8),dtype=np.float32)
    m=metrics(x,x);assert m['mse']==0 and abs(m['pixel_correlation']-1)<1e-10
    a0={'mse':1.,'psnr':0.,'v':2.};a1={'mse':3.,'psnr':0.,'v':4.}
    avg=average_seed_metrics([a0,a1]);assert avg['mse']==2. and avg['v']==3. and abs(avg['psnr']+10*math.log10(2))<1e-12
    ci=interval([2.,2.,2.],100,2);assert ci['ci95']==[2.,2.]
    undefined=interval([None,None],100,2);assert undefined['valid_clips']==0 and undefined['mean'] is None
    a.out.mkdir(parents=True,exist_ok=True);partial=a.out/'resume_fixture.jsonl';partial.write_text('{"id":"ok"}\n{"id":"broken')
    assert set(read_completed(partial))=={'ok'};assert partial.read_text()=='{"id":"ok"}\n';partial.unlink()
    report={'cpu_only':True,'noise_invariant_to_batch_partition':True,'two_seeds_differ':True,'within_clip_seed_average_not_min':True,
            'identical_prediction_metrics_correct':True,'bootstrap_constant_exact':True,'undefined_temporal_cosine_not_imputed':True,
            'interrupted_JSONL_tail_resumes':True}
    atomic_json(a.out/'self_check.json',report);print(json.dumps(report))


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    q=sub.add_parser('prepare');q.add_argument('--data',type=Path,required=True);q.add_argument('--out',type=Path,required=True);q.add_argument('--expected-clips',type=int,default=250);q.add_argument('--expected-step',type=int);q.add_argument('--checkpoint-role',choices=['final','candidate'],default='final');q.add_argument('--candidate-selection-rule',default='Externally selected proxy checkpoint; bind step and SHA before full250 inference; never choose by this full250 evaluation');q.add_argument('--size',type=int,default=128);q.add_argument('--preview-size',type=int,default=64);q.add_argument('--steps',type=int,default=16);q.add_argument('--seeds',nargs=2,type=int,default=[17,29]);q.add_argument('--bootstrap',type=int,default=10000)
    q=sub.add_parser('evaluate');q.add_argument('--data',type=Path,required=True);q.add_argument('--protocol',type=Path,required=True);q.add_argument('--checkpoint',type=Path,required=True);q.add_argument('--checkpoint-sha',required=True);q.add_argument('--pretrained',type=Path,required=True);q.add_argument('--candidate-selection-file',type=Path);q.add_argument('--out',type=Path,required=True);q.add_argument('--batch-size',type=int,default=4);q.add_argument('--max-seconds',type=int,default=7200)
    q=sub.add_parser('self-check');q.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='prepare' and (min(a.expected_clips,a.size,a.preview_size,a.steps,a.bootstrap)<1 or len(set(a.seeds))!=2):
        p.error('Positive clip/size/step/bootstrap counts and two distinct master seeds are required')
    if a.mode=='evaluate' and min(a.batch_size,a.max_seconds)<1:p.error('Batch size and time bound must be positive')
    if a.mode=='prepare':prepare(a)
    elif a.mode=='evaluate':evaluate(a)
    else:self_check(a)
if __name__=='__main__':main()
