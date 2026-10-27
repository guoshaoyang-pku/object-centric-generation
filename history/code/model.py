"""Joint visual continuation with full-volume composite-object decoding.

Future RGB supplies an autoencoding/flow target during training only. Conditioning
always receives the observed prefix alone. Flow target gradients stop at the
posterior; reconstruction and KL train the representation concurrently.
"""
import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


def cfgget(cfg, name, default):
    return cfg.get(name, default) if isinstance(cfg, dict) else getattr(cfg, name, default)


def chol(p):
    out = p.new_zeros(*p.shape[:-1], 3, 3)
    out[..., 0, 0] = p[..., 0].clamp(-7, 3).exp()
    out[..., 1, 1] = p[..., 1].clamp(-7, 3).exp()
    out[..., 2, 2] = p[..., 2].clamp(-7, 3).exp()
    out[..., 1, 0] = p[..., 3]
    out[..., 2, 0] = p[..., 4]
    out[..., 2, 1] = p[..., 5]
    return out


def shell(m):
    if m == 4:
        return torch.tensor([[-.35, 0, 0], [.35, 0, 0], [0, -.35, 0], [0, .35, 0]])
    i = torch.arange(m).float()
    z = 1 - 2*(i+.5)/m
    a = i * (math.pi*(3-math.sqrt(5)))
    r = (1-z.square()).sqrt()
    return .4 * torch.stack([r*a.cos(), r*a.sin(), z], -1)


class VisionTokens(nn.Module):
    def __init__(self, image_size, width, depth, max_frames):
        super().__init__()
        self.patch_size = 16
        self.grid = image_size // self.patch_size
        self.patch = nn.Conv2d(3, width, self.patch_size, self.patch_size)
        self.space = nn.Parameter(torch.randn(1, self.grid*self.grid, width)*.02)
        self.time = nn.Parameter(torch.randn(1, max_frames, 1, width)*.02)
        heads = max(1, width//64)
        layer = nn.TransformerEncoderLayer(width, heads, width*4, dropout=0., batch_first=True,
                                           norm_first=True, activation='gelu')
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, frames, time_start=0):
        b, t, c, h, w = frames.shape
        x = self.patch(frames.reshape(b*t, c, h, w)).flatten(2).transpose(1,2)
        if x.shape[1] != self.space.shape[1]:
            pos = self.space.transpose(1,2).reshape(1,-1,self.grid,self.grid)
            pos = F.interpolate(pos, size=(h//16,w//16), mode='bilinear', align_corners=False)
            pos = pos.flatten(2).transpose(1,2)
        else:
            pos = self.space
        x = self.blocks(x+pos)
        x = self.norm(x).reshape(b,t,-1,x.shape[-1])
        x = x + self.time[:,time_start:time_start+t]
        return x.flatten(1,2)


class ObjectPosterior(nn.Module):
    def __init__(self, objects, width, latent_dim):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1,objects,width)*.02)
        self.cross = nn.MultiheadAttention(width,max(1,width//64),batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width,width*4),nn.GELU(),nn.Linear(width*4,width))
        self.mu = nn.Linear(width,latent_dim)
        self.logvar = nn.Linear(width,latent_dim)
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias,-4.)
        self.latent_dim = latent_dim

    def forward(self, tokens, attn_bias=None):
        q = self.queries.expand(tokens.shape[0],-1,-1)
        h = q + self.cross(self.norm1(q),tokens,tokens,need_weights=False,attn_mask=attn_bias)[0]
        h = h + self.ff(self.norm2(h))
        mu = F.layer_norm(self.mu(h),(self.latent_dim,))
        logvar = self.logvar(h).clamp(-8,2)
        z = mu + (.5*logvar).exp()*torch.randn_like(mu)
        kl = .5*(mu.float().square()+logvar.float().exp()-1-logvar.float()).mean()
        return z, mu, logvar, kl


class FlowBlock(nn.Module):
    def __init__(self,width,heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width,heads,batch_first=True)
        self.cross = nn.MultiheadAttention(width,heads,batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width,width*4),nn.GELU(),nn.Linear(width*4,width))
        self.time = nn.Linear(width,width*3)

    def forward(self,x,context,t):
        a,b,c = self.time(F.silu(t)).chunk(3,-1)
        h = self.norm1(x)+a[:,None]
        x = x+self.attn(h,h,h,need_weights=False)[0]
        h = self.norm2(x)+b[:,None]
        x = x+self.cross(h,context,context,need_weights=False)[0]
        x = x+self.ff(self.norm3(x)+c[:,None])
        return x


class ObjectFlow(nn.Module):
    def __init__(self,objects,latent_dim,encoder_width,width,depth,heads):
        super().__init__()
        self.input = nn.Linear(latent_dim,width)
        self.slots = nn.Parameter(torch.randn(1,objects,width)*.02)
        self.condition = nn.Linear(encoder_width,width)
        self.time_mlp = nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))
        self.blocks = nn.ModuleList([FlowBlock(width,heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width,latent_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.width = width

    def forward(self,z,t,context):
        half = self.width//2
        f = torch.exp(-math.log(10000)*torch.arange(half,device=z.device).float()/max(half-1,1))
        phase = t.float()[:,None]*f[None]*1000
        te = torch.cat([phase.sin(),phase.cos()],-1)
        if te.shape[-1] < self.width: te = F.pad(te,(0,self.width-te.shape[-1]))
        te = self.time_mlp(te.to(z.dtype))
        x = self.input(z)+self.slots
        context = self.condition(context)
        for block in self.blocks: x = block(x,context,te)
        return self.out(self.norm(x))


class CompositeDecoder(nn.Module):
    def __init__(self,objects,latent_dim,subkernels=4,texture_kernels=64):
        super().__init__()
        self.objects,self.subkernels,self.texture_kernels = objects,subkernels,texture_kernels
        if objects==250:
            grid=(10,5,5)
        else:
            # Smoke-test and ablation grids retain deterministic xyt anchors.
            nx=max(2,round(objects**(1/3)))
            ny=max(2,round(math.sqrt(objects/nx)))
            nt=math.ceil(objects/(nx*ny));grid=(nx,ny,max(2,nt))
        nx,ny,nt=grid
        t,y,x=torch.meshgrid(torch.linspace(-1,1,nt),torch.linspace(-1,1,ny),torch.linspace(-1,1,nx),indexing='ij')
        anchors=torch.stack([x,y,t],-1).reshape(-1,3)[:objects]
        self.register_buffer('anchors',anchors)
        self.register_buffer('base_log_scale',torch.tensor([1.4/(nx-1),1.4/(ny-1),1.4/(nt-1)]).log())
        self.register_buffer('child_shell',shell(subkernels))
        g=torch.Generator().manual_seed(713)
        self.register_buffer('texture_centers',torch.randn(texture_kernels,3,generator=g)*.8)
        axis=torch.linspace(-2,2,4)
        self.register_buffer('reference',torch.stack(torch.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3))
        self.geometry_dim = 12+7*subkernels
        self.texture_dim = 12*texture_kernels
        self.output_dim = self.geometry_dim+self.texture_dim
        self.net=nn.Sequential(nn.Linear(latent_dim,256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,self.output_dim))
        nn.init.normal_(self.net[-1].weight,std=.002)
        nn.init.zeros_(self.net[-1].bias)

    def decode(self,z):
        # Renderer uses FP32 to preserve small SPD transforms under mixed precision.
        raw=self.net(z).float(); b,o,_=raw.shape; m=self.subkernels;k=self.texture_kernels
        mean=self.anchors[None]+.8*raw[...,:3].tanh()
        shape=torch.cat([self.base_log_scale[None,None]+raw[...,3:6].clamp(-2.5,2.5),.25*raw[...,6:9].tanh()],-1)
        j=9
        offset=self.child_shell[None,None]+.8*raw[...,j:j+3*m].reshape(b,o,m,3).tanh();j+=3*m
        scale=(math.log(.8)+.8*raw[...,j:j+3*m].reshape(b,o,m,3).tanh()).exp();j+=3*m
        base_amp=math.log(math.expm1(1/(1.75**3*m)))
        amplitude=F.softplus(base_amp+raw[...,j:j+m]);j+=m
        color=raw[...,j:j+3];j+=3
        out={'mean':mean,'L':chol(shape),'offset':offset,'scale':scale,'amplitude':amplitude,'color':color}
        if k:
            tex=raw[...,j:].reshape(b,o,k,12)
            tm=self.texture_centers[None,None]+.8*tex[...,:3].tanh()
            ts=torch.cat([.8*tex[...,3:6].tanh(),.4*tex[...,6:9].tanh()],-1)
            inv=torch.linalg.inv(chol(ts))
            precision=inv.transpose(-1,-2)@inv
            pm=torch.einsum('bokij,bokj->boki',precision,tm)
            packed=torch.stack([precision[...,0,0],precision[...,1,1],precision[...,2,2],precision[...,0,1],precision[...,0,2],precision[...,1,2]],-1)
            out.update(tex_p=packed,tex_pm=pm,tex_mm=(tm*pm).sum(-1),tex_rgb=.5*tex[...,9:12])
        out['inv']=torch.linalg.inv(out['L'])
        if k:
            ref=self.reference[None,:,None].expand(b,-1,o,-1)
            out['dc']=self.texture(ref,out).mean(1)
        return out

    def texture(self,z,p):
        terms=torch.stack([z[...,0]**2,z[...,1]**2,z[...,2]**2,2*z[...,0]*z[...,1],2*z[...,0]*z[...,2],2*z[...,1]*z[...,2]],-1)
        dist=torch.einsum('bqoi,boki->bqok',terms,p['tex_p'])-2*torch.einsum('bqoi,boki->bqok',z,p['tex_pm'])+p['tex_mm'][:,None]
        return torch.einsum('bqok,bokc->bqoc',(-.5*dist.clamp_min(0)).exp(),p['tex_rgb'])

    def render(self,p,query):
        z=torch.einsum('boij,bqoj->bqoi',p['inv'],query[:,:,None]-p['mean'][:,None])
        child=(z[:,:,:,None]-p['offset'][:,None])/p['scale'][:,None]
        w=((-0.5*child.square().sum(-1)).exp()*p['amplitude'][:,None]).sum(-1)
        w=w/w.sum(-1,keepdim=True).clamp_min(1e-8)
        color=p['color'][:,None]
        if self.texture_kernels: color=color+self.texture(z,p)-p['dc'][:,None]
        return (w[...,None]*color.sigmoid()).sum(2)


class JointObjectVideo(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=dict(cfg) if isinstance(cfg,dict) else vars(cfg).copy()
        self.context_frames=cfgget(cfg,'context_frames',4)
        self.future_frames=cfgget(cfg,'future_frames',20)
        self.image_size=cfgget(cfg,'image_size',128)
        self.objects=cfgget(cfg,'objects',250)
        self.latent_dim=cfgget(cfg,'latent_dim',128)
        ew=cfgget(cfg,'encoder_width',256)
        self.vision=VisionTokens(self.image_size,ew,cfgget(cfg,'encoder_depth',4),self.context_frames+self.future_frames)
        self.posterior=ObjectPosterior(self.objects,ew,self.latent_dim)
        self.flow=ObjectFlow(self.objects,self.latent_dim,ew,cfgget(cfg,'width',512),cfgget(cfg,'depth',12),cfgget(cfg,'heads',8))
        self.decoder=CompositeDecoder(self.objects,self.latent_dim,cfgget(cfg,'subkernels',4),cfgget(cfg,'texture_kernels',64))
        self.object_params_per_clip=self.objects*self.decoder.geometry_dim
        self.texture_params_per_clip=self.objects*self.decoder.texture_dim

    def parameter_counts(self):
        groups={name:sum(p.numel() for p in module.parameters()) for name,module in [('vision',self.vision),('posterior',self.posterior),('flow',self.flow),('decoder',self.decoder)]}
        groups.update(total=sum(p.numel() for p in self.parameters()),object_fields_per_clip=self.object_params_per_clip,texture_fields_per_clip=self.texture_params_per_clip,latent_values_per_clip=self.objects*self.latent_dim)
        return groups

    def forward(self,video,num_points=512):
        if video.ndim!=5 or video.shape[1] < self.context_frames+self.future_frames:
            raise ValueError('Expected full prefix+future video [B,C+T,3,H,W]')
        context=self.vision(video[:,:self.context_frames],time_start=0)
        future=video[:,self.context_frames:self.context_frames+self.future_frames]
        encoded=self.vision(future,time_start=self.context_frames)
        z,mu,logvar,kl=self.posterior(encoded)
        noise=torch.randn_like(z)
        t=torch.rand(z.shape[0],device=z.device,dtype=z.dtype)
        target=z.detach()
        zt=(1-t[:,None,None])*noise+t[:,None,None]*target
        v=self.flow(zt,t,context)
        flow=F.mse_loss(v.float(),(target-noise).float())
        b,tf,_,h,w=future.shape
        ids=torch.randint(tf*h*w,(b,num_points),device=video.device)
        ti=ids//(h*w);yi=(ids//w)%h;xi=ids%w
        query=torch.stack([2*xi.float()/max(w-1,1)-1,2*yi.float()/max(h-1,1)-1,2*ti.float()/max(tf-1,1)-1],-1)
        flat=future.permute(0,1,3,4,2).reshape(b,-1,3)
        truth=flat.gather(1,ids[...,None].expand(-1,-1,3))
        device_type=video.device.type
        with torch.autocast(device_type=device_type,enabled=False):
            p=self.decoder.decode(z.float())
            pred=torch.cat([self.decoder.render(p,q.float()) for q in query.split(256,dim=1)],dim=1)
            recon=F.mse_loss(pred,truth.float())
        return {'recon':recon,'flow':flow,'kl':kl,'latent_std':z.detach().float().std(),
                'posterior_logvar':logvar.detach().float().mean(),'sample_rgb_mean':pred.detach().mean()}

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
        _,mu,_,_=self.posterior(self.vision(future,time_start=self.context_frames))
        return self.render_latent(mu,height,width,self.future_frames)

    @torch.no_grad()
    def sample(self,context,steps=16,height=64,width=64,future_frames=None,seed=0):
        if context.shape[1]!=self.context_frames:
            raise ValueError('Sampling accepts the observed prefix only')
        features=self.vision(context,time_start=0)
        gen=torch.Generator(device=context.device).manual_seed(seed)
        z=torch.randn((context.shape[0],self.objects,self.latent_dim),device=context.device,generator=gen,dtype=features.dtype)
        for i in range(steps):
            t=torch.full((context.shape[0],),i/steps,device=context.device,dtype=z.dtype)
            z=z+self.flow(z,t,features)/steps
        return self.render_latent(z,height,width,future_frames)
