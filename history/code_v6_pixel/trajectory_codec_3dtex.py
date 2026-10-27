"""Persistent trajectory geometry with full 3-D object-local texture RBFs.

Geometry matches the frozen TrajectoryDecoder. Texture coordinates are (u,v,t):
u/v move with the parent's shared affine Bezier trajectory, while t is normalized
video time. Time-dependent appearance is not an object lifetime or identity
constraint. This independent experimental module does not modify the live codec.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .trajectory_codec import TrajectoryDecoder
    from .model import chol
except ImportError:
    from trajectory_codec import TrajectoryDecoder
    from model import chol


class Trajectory3DTextureDecoder(TrajectoryDecoder):
    """Same decode/render API; geometry43 + texture64*12 =811 fields by default.

Every texture kernel has a3-D center, six Cholesky covariance coordinates and
three RGB residual coefficients. The packed precision implementation follows
CompositeDecoder.texture, with a fixed reference DC subtraction. No alpha/depth
layer, camera model, or parent lifetime gate is introduced.
    """
    def __init__(self, objects=250, latent_dim=128, subkernels=4,
                 texture_kernels=64, temporal_support=False,
                 motion_init='static', hidden_width=256):
        # Reuse deterministic geometry buffers and exactly its initial geometry
        # output rows. The new texture head shares the original MLP trunk.
        super().__init__(objects,latent_dim,subkernels,0,temporal_support,motion_init,hidden_width)
        if texture_kernels<0:raise ValueError('texture_kernels must be nonnegative')
        self.texture_kernels=int(texture_kernels)
        self.texture_dim=12*self.texture_kernels
        self.output_dim=self.geometry_dim+self.texture_dim
        generator=torch.Generator().manual_seed(713)
        self.texture_centers=torch.randn(self.texture_kernels,3,generator=generator)*.8
        axis=torch.linspace(-2,2,4)
        self.register_buffer('texture_reference',torch.stack(torch.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3))
        if self.texture_kernels:
            geometry_head=self.net[-1]
            head=nn.Linear(hidden_width,self.output_dim)
            nn.init.normal_(head.weight,std=.002);nn.init.zeros_(head.bias)
            with torch.no_grad():
                head.weight[:self.geometry_dim].copy_(geometry_head.weight)
                head.bias[:self.geometry_dim].copy_(geometry_head.bias)
            self.net[-1]=head

    def decode(self,z):
        if z.ndim!=3 or z.shape[1:]!=(self.objects,self.latent_dim):
            raise ValueError('Expected latent shape [B, objects, latent_dim]')
        raw=self.net(z).float();b,n,_=raw.shape;m=self.subkernels;k=self.texture_kernels
        # This geometry prefix intentionally follows the frozen parent decoder
        # exactly. Motion/local-coordinate/affine methods are inherited, and a
        # same-initialization test checks every decoded geometry tensor against
        # TrajectoryDecoder(texture_kernels=0). Keeping this standalone avoids
        # modifying a source file already bound into running checkpoints.
        center=self.anchors[None]+raw[...,:2].tanh()
        offsets=1.25*raw[...,2:8].reshape(b,n,3,2).tanh()
        controls=torch.cat((center[...,None,:],center[...,None,:]+offsets),-2)
        affine_base=torch.cat((math.pi*raw[...,8:9].tanh(),
            self.base_log_scale[None,None]+raw[...,9:11].clamp(-2.5,2.5),.8*raw[...,11:12].tanh()),-1)
        affine_slope=torch.cat((math.pi*raw[...,12:13].tanh(),
            .75*raw[...,13:15].tanh(),.4*raw[...,15:16].tanh()),-1)
        j=16;children=raw[...,j:j+6*m].reshape(b,n,m,6);j+=6*m
        p={'controls':controls,'affine_base':affine_base,'affine_slope':affine_slope,
            'child_center':self.child_centers[None,None]+.8*children[...,:2].tanh(),
            'child_scale':(math.log(.65)+.8*children[...,2:4].tanh()).exp(),
            'child_angle':math.pi*children[...,4].tanh(),
            'child_log_amplitude':F.log_softmax(children[...,5],dim=-1),'color':raw[...,j:j+3]}
        j+=3
        if self.temporal_support:
            temporal=raw[...,j:j+2*m].reshape(b,n,m,2);j+=2*m
            p.update(child_time_center=temporal[...,0].tanh(),
                     child_time_scale=(math.log(1.5)+temporal[...,1].clamp(-2,1)).exp())
        if j!=self.geometry_dim:raise RuntimeError('Geometry field accounting mismatch')
        if k:
            with torch.autocast(device_type=z.device.type,enabled=False):
                tex=raw[...,j:].reshape(b,n,k,12)
                center=self.texture_centers[None,None]+.8*tex[...,:3].tanh()
                shape=torch.cat((.8*tex[...,3:6].tanh(),.4*tex[...,6:9].tanh()),-1)
                lower=chol(shape)
                inverse=torch.linalg.inv(lower)
                precision=inverse.transpose(-1,-2)@inverse
                pm=torch.einsum('bokij,bokj->boki',precision,center)
                packed=torch.stack((precision[...,0,0],precision[...,1,1],precision[...,2,2],
                                    precision[...,0,1],precision[...,0,2],precision[...,1,2]),-1)
                p.update(tex_p=packed,tex_pm=pm,tex_mm=(center*pm).sum(-1),tex_rgb=.5*tex[...,9:12],
                         tex_center=center,tex_cholesky=lower)
                ref=self.texture_reference[None,:,None].expand(b,-1,n,-1)
                p['texture_dc']=self.texture3d(ref,p).mean(1)
        return p

    @staticmethod
    def texture3d(local_uvt,p):
        """Full Gaussian texture evaluated with the legacy packed quadratic form."""
        with torch.autocast(device_type=local_uvt.device.type,enabled=False):
            z=local_uvt.float()
            terms=torch.stack((z[...,0].square(),z[...,1].square(),z[...,2].square(),
                               2*z[...,0]*z[...,1],2*z[...,0]*z[...,2],2*z[...,1]*z[...,2]),-1)
            distance=(torch.einsum('bqoi,boki->bqok',terms,p['tex_p'])
                      -2*torch.einsum('bqoi,boki->bqok',z,p['tex_pm'])+p['tex_mm'][:,None])
            return torch.einsum('bqok,bokc->bqoc',(-.5*distance.clamp_min(0)).exp(),p['tex_rgb'])

    def contributions(self,p,coords):
        """Return geometry-only parent weights and local(u,v,t) coordinates."""
        if coords.ndim!=3 or coords.shape[-1]!=3 or coords.shape[0]!=p['controls'].shape[0]:
            raise ValueError('Expected coordinates [B,Q,3] matching latent batch')
        local=self.local_coordinates(p,coords)
        delta=local[...,None,:]-p['child_center'][:,None]
        angle=p['child_angle'][:,None];c,s=angle.cos(),angle.sin();dx,dy=delta.unbind(-1)
        child=torch.stack((c*dx+s*dy,-s*dx+c*dy),-1)/p['child_scale'][:,None]
        log_density=-.5*child.square().sum(-1)+p['child_log_amplitude'][:,None]
        if self.temporal_support:
            dt=coords[...,2,None,None]-p['child_time_center'][:,None]
            log_density=log_density-.5*(dt/p['child_time_scale'][:,None]).square()
        weights=log_density.logsumexp(-1).softmax(-1)
        local_uvt=torch.cat((local,coords[...,2,None,None].expand(*local.shape[:-1],1)),-1)
        return weights,local_uvt

    def render(self,p,coords):
        with torch.autocast(device_type=coords.device.type,enabled=False):
            weights,local_uvt=self.contributions(p,coords.float())
            color=p['color'][:,None]
            if self.texture_kernels:color=color+self.texture3d(local_uvt,p)-p['texture_dc'][:,None]
            return (weights[...,None]*color.sigmoid()).sum(-2)
