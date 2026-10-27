"""Future-seen AE binding diagnostics against verified MOVi instance masks.

This evaluates reconstruction and soft parent/instance associations, not
prefix-only generation or physics. Parent contribution colors are not semantic
segmentation predictions. Zero/unassigned renderer mass is an explicit extra
parent category so numerical disappearance cannot improve entropy by omission.
"""
import argparse
import colorsys
import json
from pathlib import Path
import time

import numpy as np


def entropy(probability):
    p=np.asarray(probability,np.float64)
    positive=p>0
    return float(-(p[positive]*np.log2(p[positive])).sum())


def js_divergence(p,q):
    return max(0.,entropy((p+q)*.5)-.5*(entropy(p)+entropy(q)))


def table_metrics(counts,min_visible_pixels=16):
    """counts[T, instance including BG, parent including unassigned], pixel mass."""
    counts=np.asarray(counts,np.float64)
    if counts.ndim!=3 or not np.isfinite(counts).all() or (counts<0).any():
        raise ValueError('Expected finite nonnegative [T,I,P] joint counts')
    pooled=counts.sum(0);fg=pooled[1:];total=fg.sum()
    if total<=0:
        return {'foreground_pixel_mass':0.,'H_instance_given_parent_bits':None,
                'H_parent_given_instance_bits':None,'instances':[]}
    joint=fg/total
    pi=joint.sum(1);pp=joint.sum(0)
    h_joint=entropy(joint)
    instance_entropy=max(0.,h_joint-entropy(pp))
    fragmentation=max(0.,h_joint-entropy(pi))
    per_instance=[];adjacent=[];reappear=[]
    for label in range(1,counts.shape[1]):
        mass=counts[:,label].sum(-1)
        # Every pixel contributes total mass one, including unassigned mass.
        # Round this derived pixel count to avoid a 16-pixel object failing a
        # threshold due to floating-point summation yielding15.999999999999998.
        valid=np.flatnonzero(np.rint(mass)>=min_visible_pixels)
        distributions=counts[valid,label]/mass[valid,None] if len(valid) else np.empty((0,counts.shape[-1]))
        pairs=[]
        for j in range(1,len(valid)):
            gap=int(valid[j]-valid[j-1]);value=js_divergence(distributions[j-1],distributions[j])
            pairs.append({'from_future_frame':int(valid[j-1]),'to_future_frame':int(valid[j]),
                          'frame_gap':gap,'js_bits':value})
        near=[x['js_bits'] for x in pairs if x['frame_gap']==1]
        far=[x['js_bits'] for x in pairs if x['frame_gap']>1]
        if near:adjacent.append(float(np.mean(near)))
        if far:reappear.append(float(np.mean(far)))
        pooled_mass=pooled[label].sum()
        per_instance.append({'instance_label':label,'pixel_mass_by_frame':mass.tolist(),
            'visible_future_frames':valid.tolist(),'H_parent_given_this_instance_bits':entropy(pooled[label]/pooled_mass) if pooled_mass else None,
            'adjacent_visible_js_bits':float(np.mean(near)) if near else None,
            'reappearance_js_bits':float(np.mean(far)) if far else None,'pairs':pairs})
    all_parent_mass=pooled.sum(0)
    background_fraction=np.divide(pooled[0],all_parent_mass,out=np.zeros_like(all_parent_mass),where=all_parent_mass>0)
    bg=pooled[0];bg_total=bg.sum()
    return {'foreground_pixel_mass':float(total),
            'H_instance_given_parent_bits':instance_entropy,
            'H_parent_given_instance_bits':fragmentation,
            'effective_parents_per_instance_geometric':float(2**fragmentation),
            'foreground_weighted_parent_background_fraction':float(np.dot(pp,background_fraction)),
            'foreground_unassigned_fraction':float(fg[:,-1].sum()/total),
            'background_unassigned_fraction':float(bg[-1]/bg_total) if bg_total else None,
            'background_parent_entropy_bits':entropy(bg/bg_total) if bg_total else None,
            'adjacent_visible_js_bits_macro_instance':float(np.mean(adjacent)) if adjacent else None,
            'adjacent_valid_instances':len(adjacent),
            'reappearance_js_bits_macro_instance':float(np.mean(reappear)) if reappear else None,
            'reappearance_valid_instances':len(reappear),'instances':per_instance}


def parent_weights(decoder,params,queries):
    """Compute the same parent contribution weights as each RGB renderer."""
    import torch
    from trajectory_codec import TrajectoryDecoder
    from trajectory_codec_3dtex import Trajectory3DTextureDecoder
    with torch.autocast(device_type=queries.device.type,enabled=False):
        q=queries.float()
        if isinstance(decoder,Trajectory3DTextureDecoder):
            weights,local_uvt=decoder.contributions(params,q)
            color=params['color'][:,None]
            if decoder.texture_kernels:
                color=color+decoder.texture3d(local_uvt,params)-params['texture_dc'][:,None]
        elif isinstance(decoder,TrajectoryDecoder):
            local=decoder.local_coordinates(params,q)
            delta=local[...,None,:]-params['child_center'][:,None]
            angle=params['child_angle'][:,None];c,s=angle.cos(),angle.sin()
            dx,dy=delta.unbind(-1)
            children=torch.stack((c*dx+s*dy,-s*dx+c*dy),-1)/params['child_scale'][:,None]
            log_density=-.5*children.square().sum(-1)+params['child_log_amplitude'][:,None]
            if decoder.temporal_support:
                dt=q[...,2,None,None]-params['child_time_center'][:,None]
                log_density=log_density-.5*(dt/params['child_time_scale'][:,None]).square()
            weights=log_density.logsumexp(-1).softmax(-1)
            color=params['color'][:,None]
            if decoder.texture_kernels:
                d=(local[...,None,:]-params['texture_center'][:,None])/params['texture_scale'][:,None]
                color=color+((-.5*d.square().sum(-1)).exp()[...,None]*params['texture_rgb'][:,None]).sum(-2)
        else:
            local=torch.einsum('boij,bqoj->bqoi',params['inv'],q[:,:,None]-params['mean'][:,None])
            child=(local[:,:,:,None]-params['offset'][:,None])/params['scale'][:,None]
            density=((-.5*child.square().sum(-1)).exp()*params['amplitude'][:,None]).sum(-1)
            weights=density/density.sum(-1,keepdim=True).clamp_min(1e-8)
            color=params['color'][:,None]
            if decoder.texture_kernels:color=color+decoder.texture(local,params)-params['dc'][:,None]
        rgb=(weights[...,None]*color.sigmoid()).sum(-2)
        reference=decoder.render(params,q)
        if not torch.allclose(rgb,reference,atol=2e-6,rtol=2e-6):
            raise ValueError('Contribution implementation differs from the actual RGB renderer')
        return weights,reference


def color_labels(labels,count,background=False):
    palette=np.array([colorsys.hsv_to_rgb((i*.61803398875)%1,.65,.95) for i in range(count)],np.float32)
    if background:palette[0]=0
    else:palette[-1]=0  # Extra unassigned-parent category.
    return palette[labels]


def save_case_gif(path,truth,pred,masks,parents,parent_count):
    from PIL import Image,ImageDraw
    columns=[truth,pred,color_labels(masks,int(masks.max())+1,True),color_labels(parents,parent_count+1)]
    labels=['GT future','AE FUTURE SEEN','GT instance labels','Parent IDs (not GT)']
    frames=[];size=128;header=30
    for t in range(len(truth)):
        frame=Image.new('RGB',(size*4,size+header),'#151515');draw=ImageDraw.Draw(frame)
        for i,(video,label) in enumerate(zip(columns,labels)):
            draw.text((i*size+2,2),label,fill='white')
            draw.text((i*size+2,15),f'future frame {t}',fill='#cccccc')
            tile=Image.fromarray(np.uint8(np.clip(video[t],0,1)*255+.5))
            frame.paste(tile,(i*size,header))
        frames.append(frame)
    durations=[round((i+1)*100/12)*10-round(i*100/12)*10 for i in range(len(frames))]
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=durations,loop=0,disposal=2,optimize=False)
    frames[len(frames)//2].save(path.with_suffix('.png'))


def run(a):
    import torch
    from codec_research import Codec,CacheBank,canonical_sha,digest,atomic_json
    if a.out.exists() and any(a.out.iterdir()):raise ValueError('Binding evaluation needs a new/empty output directory')
    started=time.time();deadline=min(started+a.max_hours*3600,a.deadline_unix or float('inf'))
    a.out.mkdir(parents=True,exist_ok=True)
    bank=CacheBank(a.data,a.cache)
    checkpoint_sha=digest(a.checkpoint)
    if a.checkpoint_sha and checkpoint_sha!=a.checkpoint_sha:raise ValueError('Checkpoint SHA mismatch')
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if ckpt.get('format')!='codec_research_v1':raise ValueError('Requires codec_research checkpoint')
    bound=ckpt.get('protocol')
    if not isinstance(bound,dict) or canonical_sha(bound)!=ckpt['protocol_sha256']:
        raise ValueError('Checkpoint lacks valid embedded protocol')
    for key in ['model_config','source_sha256','data_manifest_sha256','cache_protocol_sha256']:
        if bound[key]!=ckpt[key]:raise ValueError('Checkpoint binding mismatch: '+key)
    if bound['data_manifest_sha256']!=bank.manifest_sha or bound['cache_protocol_sha256']!=bank.protocol_sha:
        raise ValueError('Evaluation data/cache differs from training contract')
    for name,sha in bound['source_sha256'].items():
        if Path(name).name!=name or digest(Path(__file__).with_name(name))!=sha:
            raise ValueError('Training source has changed: '+name)
    gt_manifest=json.loads((a.gt/'manifest.json').read_text())
    if gt_manifest.get('status')!='complete':raise ValueError('GT annotation extraction is incomplete')
    gt_lookup={r['id']:r for r in gt_manifest['records']}
    ids=a.ids if a.ids else list(gt_lookup)
    if not ids or len(set(ids))!=len(ids) or any(i not in gt_lookup for i in ids):raise ValueError('Invalid GT IDs')
    device=torch.device(a.device);torch.set_num_threads(4)
    if device.type=='cuda':torch.cuda.set_device(device)
    model=Codec(**ckpt['model_config']).to(device).eval();model.load_state_dict(ckpt['model'],strict=True)
    model.assert_decoder_unchanged(bound.get('decoder_initial_sha256'))
    protocol={'version':'future-ae-object-binding-v1','checkpoint_sha256':checkpoint_sha,'step':ckpt['step'],
        'variant':model.variant,'training_protocol_sha256':ckpt['protocol_sha256'],'data_manifest_sha256':bank.manifest_sha,
        'cache_protocol_sha256':bank.protocol_sha,'annotation_manifest_sha256':digest(a.gt/'manifest.json'),
        'evaluator_sha256':digest(Path(__file__)),'ids':ids,'native_metric_size':128,'future_slice':[4,24],
        'min_visible_pixels':a.min_visible_pixels,'target_seen':True,'amp':a.amp,'chunk':a.chunk,
        'definition':{'joint_table':'C[t,instance,parent] is sum of actual normalized parent weights at GT instance pixels; label0 is background; final parent is unassigned renderer mass',
          'purity':'H(instance|parent), pooled over foreground pixels and time, bits; smaller means purer, not semantic-label accuracy',
          'fragmentation':'H(parent|instance), pooled over foreground pixels and time, bits; smaller means fewer effective parents; multiple parents may be needed for a complex object',
          'temporal_js':'p(parent|instance,t) normalizes contributions over visible pixels; macro mean across instances of consecutive-frame JS in bits; visibility is mask pixel count, not source visibility field',
          'reappearance_js':'JS between successive qualifying visible frames separated by a visibility gap; reported separately from adjacent frames',
          'background_fraction':'Foreground-weighted mean p(background|parent); exposes parents that mix background and foreground',
          'mse':'Mean RGB squared error on GT foreground/background pixels separately, before GIF quantization'},
        'scope':'Future-seen AE binding diagnostic, not continuation, semantic masks, physics, or a blind test. Entropies can be gamed by coarse rendering; interpret jointly with reconstruction error and masks.'}
    atomic_json(a.out/'protocol.json',protocol);results=[]
    axis=torch.linspace(-1,1,128,device=device);tt=torch.linspace(-1,1,20,device=device)
    tq,yq,xq=torch.meshgrid(tt,axis,axis,indexing='ij');queries=torch.stack((xq,yq,tq),-1).reshape(1,-1,3)
    try:
        with torch.inference_mode():
            for identifier in ids:
                if time.time()>=deadline:raise TimeoutError('Binding evaluation deadline reached')
                record=gt_lookup[identifier];source=bank.records[identifier]
                if (not record.get('source_rgb_exactly_equal') or record['rgb_array_sha256']!=source['array_sha256'] or
                    record['source_record_sha256']!=source['source_record_sha256'] or record['source_url']!=source['source_url']):
                    raise ValueError('GT source does not match RGB: '+identifier)
                npz_path=a.gt/identifier/'annotations.npz';mask_path=a.gt/identifier/'segmentations.npy'
                if digest(npz_path)!=record['annotations_sha256'] or digest(mask_path)!=record['segmentations_npy_sha256']:
                    raise ValueError('GT artifact checksum mismatch: '+identifier)
                masks=np.load(mask_path,allow_pickle=False)
                with np.load(npz_path,allow_pickle=False) as gt:
                    if not np.array_equal(masks,gt['segmentations']):raise ValueError('Two GT mask copies differ')
                if masks.shape!=(24,128,128) or masks.dtype!=np.uint8 or masks.max()>record['num_instances']:
                    raise ValueError('Unexpected GT mask contract')
                masks=masks[4:24];flat_labels=masks.reshape(-1)
                truth=np.array(bank.rgb(identifier)[4:24],dtype=np.float32)/255
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=a.amp and device.type=='cuda'):
                    _,mu,_,_=model.encode_future(*bank.batch([identifier],device))
                params=model.decoder.decode(mu.float())
                counts=np.zeros((20,record['num_instances']+1,model.objects+1),np.float64)
                predicted=np.empty((20*128*128,3),np.float32);parent_ids=np.empty(20*128*128,np.uint16)
                cursor=0
                for query in queries.split(a.chunk,dim=1):
                    if time.time()>=deadline:raise TimeoutError('Binding evaluation deadline reached')
                    weights,rgb=parent_weights(model.decoder,params,query)
                    w=weights[0].float().cpu().numpy().astype(np.float64);n=len(w)
                    if not np.isfinite(w).all() or (w<0).any() or (w.sum(-1)>1.00001).any():raise ValueError('Invalid parent weights')
                    # Compensate tiny floating-point excess; substantial missing
                    # mass gets its own category instead of being discarded.
                    w=w/np.maximum(w.sum(-1,keepdims=True),1.)
                    w=np.concatenate((w,np.maximum(0.,1-w.sum(-1,keepdims=True))),axis=-1)
                    frame_indices=np.arange(cursor,cursor+n)//(128*128)
                    np.add.at(counts,(frame_indices,flat_labels[cursor:cursor+n]),w)
                    predicted[cursor:cursor+n]=rgb[0].float().cpu().numpy();parent_ids[cursor:cursor+n]=w.argmax(-1)
                    cursor+=n
                predicted=predicted.reshape(20,128,128,3);parent_ids=parent_ids.reshape(20,128,128)
                if not np.isfinite(predicted).all():raise ValueError('Nonfinite reconstruction')
                squared=np.square(predicted.astype(np.float64)-truth)
                foreground=masks>0;background=~foreground
                metrics=table_metrics(counts,a.min_visible_pixels)
                metrics.update(foreground_mse=float(squared[foreground].mean()) if foreground.any() else None,
                    background_mse=float(squared[background].mean()) if background.any() else None,
                    all_pixel_mse=float(squared.mean()),foreground_pixels=int(foreground.sum()),background_pixels=int(background.sum()))
                result={'id':identifier,'split':source['split'],'target_seen':True,'step':ckpt['step'],
                        'source_array_sha256':source['array_sha256'],'annotation_sha256':record['annotations_sha256'],'metrics':metrics}
                atomic_json(a.out/(identifier+'.json'),result)
                np.savez_compressed(a.out/(identifier+'_binding.npz'),joint_counts=counts.astype(np.float32),parent_argmax=parent_ids)
                save_case_gif(a.out/(identifier+'.gif'),truth,predicted,masks,parent_ids,model.objects)
                results.append(result)
                atomic_json(a.out/'progress.json',{'completed':len(results),'total':len(ids),'elapsed_seconds':time.time()-started})
                print(json.dumps({'id':identifier,'completed':len(results),'elapsed_seconds':time.time()-started,
                    'foreground_mse':metrics['foreground_mse'],'fragmentation_bits':metrics['H_parent_given_instance_bits']}),flush=True)
        scalar_keys=['foreground_mse','background_mse','all_pixel_mse','H_instance_given_parent_bits','H_parent_given_instance_bits',
                     'foreground_weighted_parent_background_fraction','foreground_unassigned_fraction',
                     'adjacent_visible_js_bits_macro_instance','reappearance_js_bits_macro_instance']
        aggregates={}
        for split in ['train','validation']:
            chosen=[r for r in results if r['split']==split]
            if not chosen:continue
            aggregates[split]={'clips':len(chosen),'metrics':{}}
            for key in scalar_keys:
                values=[r['metrics'].get(key) for r in chosen if r['metrics'].get(key) is not None]
                aggregates[split]['metrics'][key]={'mean':float(np.mean(values)) if values else None,'valid_clips':len(values)}
        atomic_json(a.out/'summary.json',{'status':'complete','protocol':protocol,'records':results,'by_split':aggregates,
            'elapsed_seconds':time.time()-started,'note':'Clip averages; no statistical superiority/physics claim from these eight selected cases.'})
        atomic_json(a.out/'checksums.json',{p.name:digest(p) for p in sorted(a.out.iterdir()) if p.is_file() and p.name!='checksums.json'})
    except Exception as exc:
        atomic_json(a.out/'error.json',{'error':type(exc).__name__,'detail':str(exc),'completed':len(results),'elapsed_seconds':time.time()-started});raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','data','cache','gt','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--checkpoint-sha');p.add_argument('--ids',nargs='+');p.add_argument('--chunk',type=int,default=256)
    p.add_argument('--min-visible-pixels',type=int,default=16);p.add_argument('--max-hours',type=float,default=1)
    p.add_argument('--deadline-unix',type=float);p.add_argument('--device',default='cuda:0')
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    a=p.parse_args()
    if min(a.chunk,a.min_visible_pixels,a.max_hours)<=0:p.error('Bounds must be positive')
    run(a)


if __name__=='__main__':main()
