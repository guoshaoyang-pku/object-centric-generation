"""Raw 64px RGB target: exact patches and fixed train-only standardization.

There is no learned target encoder/decoder and no VAE target. The frozen VAE
features used by the observed-prefix condition are an independent input path.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from codec_research import canonical_sha, digest
from small_object_flow import SmallObjectFlow

RAW_PREPROCESS = {'size':64,'frames':20,'offset':4,'input_layout':'THWC','output_layout':'TCHW',
    'input_dtype':'uint8','resize':'bilinear','align_corners':False,'antialias':False,'scale':255.0,
    'compute_dtype':'float32','accumulate_dtype':'float64'}
TARGET_CONTRACT = {'target_kind':'raw future RGB; train-only global channel mean/std',
    'video_t_c_h_w':[20,3,64,64], 'patch_t_h_w':[1,8,8], 'tokens':1280,'token_width':192,
    'latent_scalars':245760,'packed_order':'t,patch_y,patch_x,dy,dx,channel',
    'preprocess':RAW_PREPROCESS,'decoder':'fixed channel unnormalize and inverse patch permutation; clamp01 only for display/metrics',
    'claim_limit':'Raw-pixel flow at common64; target length/capacity differ from object and VAE patch'}
STATS_SOURCE_SHA = '3459521c5135458b5446a87805b3b01a78cfa1665240b8677f0c728fd56a25ed'


def preprocess_rgb(array):
    if array.shape!=(24,128,128,3) or array.dtype!=np.uint8:
        raise ValueError('Require original uint8 RGB[24,128,128,3]')
    video=torch.from_numpy(np.array(array[4:24],copy=True)).permute(0,3,1,2).float()/255.
    return F.interpolate(video,(64,64),mode='bilinear',align_corners=False,antialias=False)


def pack_rgb(video):
    if video.ndim!=5 or video.shape[1:]!=(20,3,64,64):raise ValueError('Expected [B,20,3,64,64]')
    grid=video.permute(0,1,3,4,2).reshape(-1,20,8,8,8,8,3)
    return grid.permute(0,1,2,4,3,5,6).reshape(-1,1280,192)


def unpack_rgb(tokens):
    if tokens.ndim!=3 or tokens.shape[1:]!=(1280,192):raise ValueError('Expected [B,1280,192]')
    grid=tokens.reshape(-1,20,8,8,8,8,3).permute(0,1,2,4,3,5,6)
    return grid.reshape(-1,20,64,64,3).permute(0,1,4,2,3)


def patch_anchors():
    axis=torch.linspace(-1,1,64).reshape(8,8).mean(1)
    t,y,x=torch.meshgrid(torch.linspace(-1,1,20),axis,axis,indexing='ij')
    return torch.stack((x,y,t),-1).reshape(1280,3)


def load_statistics(path, expected_file_sha, bank):
    if not expected_file_sha or digest(path)!=expected_file_sha:raise ValueError('Raw statistics file SHA mismatch')
    statistics=json.loads(Path(path).read_text());body={k:v for k,v in statistics.items() if k!='stats_sha256'}
    if canonical_sha(body)!=statistics.get('stats_sha256'):raise ValueError('Raw statistics canonical SHA mismatch')
    if statistics.get('version')!='raw-pixel-train-statistics-v1' or statistics.get('raw_target_contract')!=RAW_PREPROCESS:
        raise ValueError('Raw statistics preprocessing differs')
    if statistics['source_code_sha256']!=STATS_SOURCE_SHA:raise ValueError('Raw statistics source implementation differs')
    if statistics['data_manifest_sha256']!=bank.manifest_sha:raise ValueError('Raw statistics data manifest differs')
    expected=[{k:bank.records[i][k] for k in ['id','split','array','array_sha256','shape','dtype']} for i in bank.ids('train')]
    if statistics['records']!=expected or statistics['count']!=len(expected):
        raise ValueError('Raw statistics must cover exactly all train records and no validation')
    pixels=len(expected)*20*64*64
    if statistics['pixels_per_channel']!=pixels:raise ValueError('Raw statistics population count differs')
    mean=torch.tensor(statistics['channel_mean'],dtype=torch.float64);std=torch.tensor(statistics['channel_std'],dtype=torch.float64)
    if mean.shape!=(3,) or std.shape!=(3,) or not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std>0).all()):
        raise ValueError('Invalid raw channel normalization')
    derived_mean=torch.tensor(statistics['channel_sum'],dtype=torch.float64)/pixels
    derived_std=(torch.tensor(statistics['channel_sum_squares'],dtype=torch.float64)/pixels-derived_mean.square()).sqrt()
    if not torch.allclose(mean,derived_mean,rtol=0,atol=1e-14) or not torch.allclose(std,derived_std,rtol=0,atol=1e-14):
        raise ValueError('Raw statistics moments differ from supplied mean/std')
    return statistics,{'file_sha256':expected_file_sha,'stats_sha256':statistics['stats_sha256'],
        'data_manifest_sha256':bank.manifest_sha,'count':len(expected),'pixels_per_channel':pixels,
        'channel_mean':statistics['channel_mean'],'channel_std':statistics['channel_std'],
        'source_code_sha256':statistics['source_code_sha256'],'scope':statistics['scope']}


class RawPixelFlow(SmallObjectFlow):
    def __init__(self,encoder_width=256,width=384,depth=8,heads=6):
        super().__init__(1280,192,encoder_width,patch_anchors(),width,depth,heads)
    def get_extra_state(self):
        result=super().get_extra_state();result.update(backend='raw_pixel64',target_contract=TARGET_CONTRACT);return result


class FixedPixelDecoder(nn.Module):
    def __init__(self,mean,std):
        super().__init__()
        self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32).reshape(1,1,3,1,1))
        self.register_buffer('std',torch.tensor(std,dtype=torch.float32).reshape(1,1,3,1,1))
        if not bool(torch.isfinite(self.mean).all() and torch.isfinite(self.std).all() and (self.std>0).all()):raise ValueError('Invalid normalization')
        self.eval()
    def train(self,mode=True):return super().train(False)
    @torch.no_grad()
    def encode_rgb(self,video):return pack_rgb((video.float()-self.mean)/self.std)
    @torch.no_grad()
    def decode_tokens(self,tokens,size=64):
        if size!=64:raise ValueError('Raw pixel baseline is native64 only; do not compare it as native128')
        return (unpack_rgb(tokens.float())*self.std+self.mean).clamp(0,1)
