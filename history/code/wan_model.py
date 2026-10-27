"""Object-volume continuation adapted from a pretrained Wan video generator.

Wan's VAE and original transformer weights stay frozen. Trainable LoRA layers,
visual-conditioning/object adapters, the object posterior and splat decoder are
jointly optimized. This changes Wan's latent interface: pretrained video quality
is not presumed to transfer without adaptation.
"""
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

try:
    from .model import CompositeDecoder, ObjectPosterior, cfgget
except ImportError:
    from model import CompositeDecoder, ObjectPosterior, cfgget


def position_features(coords, width):
    """Fixed XYZ features shared by VAE feature positions and object queries."""
    bands=max(1,width//6)
    frequency=torch.exp(torch.linspace(0,math.log(16),bands,device=coords.device,dtype=torch.float32))*math.pi
    phase=coords.float()[...,None]*frequency
    encoded=torch.cat([phase.sin(),phase.cos()],-1).flatten(-2)
    return F.pad(encoded,(0,max(0,width-encoded.shape[-1])))[...,:width]


class LoRALinear(nn.Module):
    def __init__(self,base,rank=8,alpha=8.):
        super().__init__()
        self.base=base
        self.base.requires_grad_(False)
        self.in_features,self.out_features=base.in_features,base.out_features
        self.rank=rank
        self.scale=alpha/rank
        self.lora_A=nn.Parameter(torch.empty(rank,base.in_features,dtype=torch.float32,device=base.weight.device))
        self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank,dtype=torch.float32,device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))

    def forward(self,x):
        original=self.base(x)
        with torch.autocast(device_type=x.device.type,enabled=False):
            residual=F.linear(F.linear(x.float(),self.lora_A),self.lora_B)*self.scale
        return original+residual.to(original.dtype)


def add_lora(backbone,rank,alpha):
    selected=[]
    for name,module in list(backbone.named_modules()):
        if not isinstance(module,nn.Linear):
            continue
        parts=name.split('.')
        if not any(p in {'attn1','attn2'} for p in parts):
            continue
        if not (name.endswith(('to_q','to_k','to_v','to_out.0'))):
            continue
        parent_name,leaf=name.rsplit('.',1)
        parent=backbone.get_submodule(parent_name)
        replacement=LoRALinear(module,rank,alpha)
        if leaf.isdigit():
            parent[int(leaf)]=replacement
        else:
            setattr(parent,leaf,replacement)
        selected.append(name)
    if not selected:
        raise ValueError('No Wan attention q/k/v/out projections found; refusing an unadapted backbone')
    return selected


class WanObjectFlow(nn.Module):
    def __init__(self,backbone,latent_dim,feature_width,objects,rank=8,alpha=8.):
        super().__init__()
        if objects!=250:
            raise ValueError('Pretrained Wan object adapter currently requires the 5x5x10=250 object grid')
        self.backbone=backbone
        self.backbone.requires_grad_(False)
        self.channels=int(backbone.config.in_channels)
        self.patch_size=tuple(backbone.config.patch_size)
        if self.patch_size!=(1,2,2):
            raise ValueError(f'Expected Wan patch size (1,2,2), got {self.patch_size}')
        self.grid=(5,5,10)
        fields=self.channels*math.prod(self.patch_size)
        self.input=nn.Linear(latent_dim,fields)
        self.output=nn.Linear(fields,latent_dim)
        self.condition=nn.Sequential(nn.Linear(feature_width,int(backbone.config.text_dim)),nn.LayerNorm(int(backbone.config.text_dim)))
        self.lora_modules=add_lora(backbone,rank,alpha)
        if hasattr(backbone,'enable_gradient_checkpointing'):
            backbone.enable_gradient_checkpointing()
        # Small trainable output stabilizes the newly introduced object interface.
        nn.init.normal_(self.output.weight,std=.005)
        nn.init.zeros_(self.output.bias)

    def unpatchify(self,tokens):
        b=tokens.shape[0];t,h,w=self.grid;pt,ph,pw=self.patch_size
        x=tokens.reshape(b,t,h,w,pt,ph,pw,self.channels)
        return x.permute(0,7,1,4,2,5,3,6).flatten(6,7).flatten(4,5).flatten(2,3)

    def patchify(self,volume):
        b=volume.shape[0];t,h,w=self.grid;pt,ph,pw=self.patch_size
        return volume.reshape(b,self.channels,t,pt,h,ph,w,pw).permute(0,2,4,6,3,5,7,1).reshape(b,t*h*w,-1)

    def forward(self,z,t,context):
        use_amp=z.device.type=='cuda'
        with torch.autocast(device_type=z.device.type,dtype=torch.bfloat16,enabled=use_amp):
            volume=self.unpatchify(self.input(z))
            condition=self.condition(context)
            # Native Wan time denotes noise level; our FM time runs noise(0)->clean(1).
            native_timestep=(1-t.float())*1000
            predicted=self.backbone(hidden_states=volume,encoder_hidden_states=condition,
                                    timestep=native_timestep,return_dict=False)[0]
            return self.output(self.patchify(predicted))


class PretrainedJointObjectVideo(nn.Module):
    def __init__(self,cfg,vae=None,backbone=None):
        super().__init__()
        self.cfg=dict(cfg) if isinstance(cfg,dict) else vars(cfg).copy()
        self.context_frames=int(cfgget(cfg,'context_frames',4))
        self.future_frames=int(cfgget(cfg,'future_frames',20))
        self.image_size=int(cfgget(cfg,'image_size',128))
        self.objects=int(cfgget(cfg,'objects',250))
        self.latent_dim=int(cfgget(cfg,'latent_dim',128))
        self.posterior_locality=bool(cfgget(cfg,'posterior_locality',True))
        self.posterior_sigma=tuple(float(x) for x in cfgget(cfg,'posterior_sigma',(.4,.5,.5)))
        if len(self.posterior_sigma)!=3 or min(self.posterior_sigma)<=0:
            raise ValueError('posterior_sigma must have three positive x/y/t scales')
        self.cfg.update(posterior_locality=self.posterior_locality,posterior_sigma=list(self.posterior_sigma))
        feature_width=int(cfgget(cfg,'encoder_width',256))
        if vae is None or backbone is None:
            from diffusers import AutoencoderKLWan,WanTransformer3DModel
            path=cfgget(cfg,'pretrained_path',None)
            if not path:
                raise ValueError('pretrained_path is required: scratch fallback is deliberately disabled')
            if vae is None:
                vae=AutoencoderKLWan.from_pretrained(path,subfolder='vae',torch_dtype=torch.float32,local_files_only=True)
            if backbone is None:
                backbone=WanTransformer3DModel.from_pretrained(path,subfolder='transformer',torch_dtype=torch.bfloat16,local_files_only=True)
        self.vae=vae.float().eval()
        self.vae.requires_grad_(False)
        channels=int(self.vae.config.z_dim)
        if channels!=int(backbone.config.in_channels):
            raise ValueError('Wan VAE/transformer channel mismatch')
        self.register_buffer('latent_mean',torch.tensor(self.vae.config.latents_mean,dtype=torch.float32).reshape(1,channels,1,1,1))
        self.register_buffer('latent_std',torch.tensor(self.vae.config.latents_std,dtype=torch.float32).reshape(1,channels,1,1,1))
        self.visual_adapter=nn.Sequential(nn.Linear(channels,feature_width),nn.LayerNorm(feature_width),nn.GELU(),nn.Linear(feature_width,feature_width))
        self.posterior=ObjectPosterior(self.objects,feature_width,self.latent_dim)
        self.decoder=CompositeDecoder(self.objects,self.latent_dim,int(cfgget(cfg,'subkernels',4)),int(cfgget(cfg,'texture_kernels',64)))
        with torch.no_grad():
            self.posterior.queries.add_(.1*position_features(self.decoder.anchors,feature_width)[None])
        self.flow=WanObjectFlow(backbone,self.latent_dim,feature_width,self.objects,
                                int(cfgget(cfg,'lora_rank',8)),float(cfgget(cfg,'lora_alpha',8.)))
        self.object_params_per_clip=self.objects*self.decoder.geometry_dim
        self.texture_params_per_clip=self.objects*self.decoder.texture_dim
        self.pretrained_info={'model':'Wan-AI/Wan2.2-TI2V-5B-Diffusers','path':str(cfgget(cfg,'pretrained_path','injected-test-components')),
            'vae':'frozen float32','transformer_base':'frozen bfloat16','lora_rank':int(cfgget(cfg,'lora_rank',8)),
            'lora_alpha':float(cfgget(cfg,'lora_alpha',8.)),'lora_projection_count':len(self.flow.lora_modules),
            'native_patch_size':list(self.flow.patch_size),'object_grid':list(self.flow.grid),
            'posterior_locality':self.posterior_locality,'posterior_sigma_xyz':list(self.posterior_sigma),
            'target_gradient_isolation':'future posterior target and prefix shared-adapter features detached for flow loss',
            'note':'New object latent and visual condition adapters; this is pretrained initialization adaptation, not an unchanged Wan inference pipeline'}

    def train(self,mode=True):
        super().train(mode)
        self.vae.eval()
        return self

    @torch.no_grad()
    def _encode_vae(self,frames):
        # Wan's causal temporal VAE expects 4k+1 frames. Padding never reads future context.
        x=frames.permute(0,2,1,3,4).float()*2-1
        factor=int(getattr(self.vae.config,'scale_factor_temporal',4))
        pad=(factor-(x.shape[2]-1)%factor)%factor
        if pad:
            x=torch.cat([x,x[:,:,-1:].expand(-1,-1,pad,-1,-1)],dim=2)
        with torch.autocast(device_type=x.device.type,enabled=False):
            result=self.vae.encode(x,return_dict=True)
            latent=result.latent_dist.mode()
            return (latent-self.latent_mean)/self.latent_std.clamp_min(1e-6)

    def visual_features(self,frames,time_start=0,return_coords=False):
        latent=self._encode_vae(frames)
        b,c,t,h,w=latent.shape
        zz,yy,xx=torch.meshgrid(torch.linspace(0,max(frames.shape[1]-1,1),t,device=latent.device),
                                torch.linspace(-1,1,h,device=latent.device),torch.linspace(-1,1,w,device=latent.device),indexing='ij')
        total=max(self.context_frames+self.future_frames-1,1)
        xyz=torch.stack([xx,yy,2*(zz+time_start)/total-1],-1).reshape(-1,3)
        tokens=latent.flatten(2).transpose(1,2)
        adapted=self.visual_adapter(tokens)
        features=adapted+.1*position_features(xyz,adapted.shape[-1])[None].to(adapted.dtype)
        if return_coords:
            # Object anchors describe the future window's own [-1,1] time,
            # distinct from the absolute prefix+future positional feature above.
            local_xyz=torch.stack([xx,yy,2*zz/max(frames.shape[1]-1,1)-1],-1).reshape(-1,3)
            return features,local_xyz
        return features

    def _future_posterior(self,future):
        features,xyz=self.visual_features(future,time_start=self.context_frames,return_coords=True)
        bias=None
        if self.posterior_locality:
            sigma=xyz.new_tensor(self.posterior_sigma)
            delta=(self.decoder.anchors[:,None]-xyz[None])/sigma
            # A finite soft preference, never a hard attention mask. It only
            # anchors the reconstruction pool; Wan's generation remains global.
            bias=-.5*delta.square().sum(-1)
        return self.posterior(features,attn_bias=bias)

    def parameter_counts(self):
        groups={name:sum(p.numel() for p in module.parameters()) for name,module in [('vae',self.vae),('visual_adapter',self.visual_adapter),('posterior',self.posterior),('flow',self.flow),('decoder',self.decoder)]}
        groups.update(total=sum(p.numel() for p in self.parameters()),trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
                      frozen=sum(p.numel() for p in self.parameters() if not p.requires_grad),
                      lora=sum(p.numel() for n,p in self.named_parameters() if 'lora_A' in n or 'lora_B' in n),
                      object_fields_per_clip=self.object_params_per_clip,texture_fields_per_clip=self.texture_params_per_clip,
                      latent_values_per_clip=self.objects*self.latent_dim)
        return groups

    def rollout_latent(self,context,initial_noise,steps=4,checkpoint_steps=True):
        """Differentiable Euler rollout; arguments contain no future targets.

        Checkpoint each entire velocity call (Wan also checkpoints its blocks).
        No state is detached between steps. Context follows the same detached
        shared-adapter policy as the latent flow training branch.
        """
        if steps<1:
            raise ValueError('Rollout steps must be positive')
        z=initial_noise
        for index in range(steps):
            t=torch.full((z.shape[0],),index/steps,device=z.device,dtype=z.dtype)
            if checkpoint_steps and torch.is_grad_enabled():
                velocity=checkpoint(self.flow,z,t,context,use_reentrant=False,preserve_rng_state=True)
            else:
                velocity=self.flow(z,t,context)
            z=z+velocity/steps
        return z

    def forward(self,video,num_points=512,generated_rgb_aux=False,rollout_rgb_aux=False,rollout_steps=4):
        if video.ndim!=5 or video.shape[1]<self.context_frames+self.future_frames:
            raise ValueError('Expected prefix+future RGB [B,C+T,3,H,W]')
        # The adapter also encodes future targets. Detach the prefix branch so
        # flow conditioning cannot rotate that shared target representation;
        # flow.condition and LoRA still receive their normal training gradients.
        context=self.visual_features(video[:,:self.context_frames],time_start=0).detach()
        # Draw from pure noise before even constructing the future posterior.
        rollout_noise=None
        if rollout_rgb_aux:
            rollout_noise=torch.randn((video.shape[0],self.objects,self.latent_dim),
                                      device=context.device,dtype=context.dtype)
        future=video[:,self.context_frames:self.context_frames+self.future_frames]
        z,mu,logvar,kl=self._future_posterior(future)
        noise=torch.randn_like(z);t=torch.rand(z.shape[0],device=z.device,dtype=z.dtype)
        target=z.detach();zt=(1-t[:,None,None])*noise+t[:,None,None]*target
        velocity=self.flow(zt,t,context)
        flow=F.mse_loss(velocity.float(),(target-noise).float())
        b,tf,_,h,w=future.shape
        ids=torch.randint(tf*h*w,(b,num_points),device=video.device)
        ti=ids//(h*w);yi=(ids//w)%h;xi=ids%w
        query=torch.stack([2*xi.float()/max(w-1,1)-1,2*yi.float()/max(h-1,1)-1,2*ti.float()/max(tf-1,1)-1],-1)
        flat=future.permute(0,1,3,4,2).reshape(b,-1,3)
        truth=flat.gather(1,ids[...,None].expand(-1,-1,3))
        with torch.autocast(device_type=video.device.type,enabled=False):
            p=self.decoder.decode(z.float())
            predicted=torch.cat([self.decoder.render(p,q.float()) for q in query.split(256,dim=1)],dim=1)
            recon=F.mse_loss(predicted,truth.float())
            generated_rgb=recon.new_zeros(())
            generated_rgb_unweighted=recon.new_zeros(())
            if generated_rgb_aux:
                # Teacher-noised endpoint estimate, NOT a prefix-only rollout:
                # zt includes detached future information whenever t > 0.
                # Linear t weighting reduces pressure to average ambiguous
                # futures at high noise. It does not eliminate that risk.
                estimate=zt.float()+(1-t.float()[:,None,None])*velocity.float()
                generated_params=self.decoder.decode(estimate)
                generated=torch.cat([self.decoder.render(generated_params,q.float())
                                     for q in query.split(256,dim=1)],dim=1)
                per_clip=(generated-truth.float()).square().mean(dim=(1,2))
                generated_rgb=(t.float()*per_clip).mean()
                generated_rgb_unweighted=per_clip.mean()
        rollout_rgb=recon.new_zeros(())
        if rollout_rgb_aux:
            # Future pixels enter only as labels after the full pure-noise
            # prefix-conditioned rollout. RGB MSE may still favor mean futures.
            rollout_z=self.rollout_latent(context,rollout_noise,steps=rollout_steps)
            with torch.autocast(device_type=video.device.type,enabled=False):
                rollout_params=self.decoder.decode(rollout_z.float())
                rollout_pixels=torch.cat([self.decoder.render(rollout_params,q.float())
                                          for q in query.split(256,dim=1)],dim=1)
                rollout_rgb=F.mse_loss(rollout_pixels,truth.float())
        return {'recon':recon,'flow':flow,'kl':kl,'latent_std':z.detach().float().std(),
                'generated_rgb':generated_rgb,'generated_rgb_unweighted':generated_rgb_unweighted,
                'rollout_rgb':rollout_rgb,
                'posterior_logvar':logvar.detach().float().mean(),'sample_rgb_mean':predicted.detach().mean()}

    @torch.no_grad()
    def render_latent(self,z,height=64,width=64,future_frames=None):
        t=self.future_frames if future_frames is None else future_frames
        tt,yy,xx=torch.meshgrid(torch.linspace(-1,1,t,device=z.device),torch.linspace(-1,1,height,device=z.device),torch.linspace(-1,1,width,device=z.device),indexing='ij')
        q=torch.stack([xx,yy,tt],-1).reshape(1,-1,3).expand(z.shape[0],-1,-1)
        with torch.autocast(device_type=z.device.type,enabled=False):
            p=self.decoder.decode(z.float())
            pixels=torch.cat([self.decoder.render(p,s.float()) for s in q.split(512,dim=1)],dim=1)
        return pixels.reshape(z.shape[0],t,height,width,3).permute(0,1,4,2,3).clamp(0,1)

    @torch.no_grad()
    def reconstruct(self,video,height=64,width=64):
        future=video[:,self.context_frames:self.context_frames+self.future_frames]
        _,mu,_,_=self._future_posterior(future)
        return self.render_latent(mu,height,width,self.future_frames)

    @torch.no_grad()
    def sample(self,context,steps=16,height=64,width=64,future_frames=None,seed=0):
        if context.shape[1]!=self.context_frames:
            raise ValueError('Sampling accepts observed prefix only')
        features=self.visual_features(context,time_start=0)
        generator=torch.Generator(device=context.device).manual_seed(seed)
        z=torch.randn((context.shape[0],self.objects,self.latent_dim),device=context.device,generator=generator,dtype=features.dtype)
        for i in range(steps):
            t=torch.full((context.shape[0],),i/steps,device=context.device,dtype=z.dtype)
            z=z+self.flow(z,t,features)/steps
        return self.render_latent(z,height,width,future_frames)
