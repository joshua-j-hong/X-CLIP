from collections import OrderedDict
from typing import Tuple, Union
import hashlib, os, urllib, warnings
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn

_MODELS = {
    "RN50":"https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101":"https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4":"https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16":"https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32":"https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16":"https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}
_PT_NAME={"RN50":"RN50.pt","RN101":"RN101.pt","RN50x4":"RN50x4.pt","RN50x16":"RN50x16.pt","ViT-B/32":"ViT-B-32.pt","ViT-B/16":"ViT-B-16.pt"}

def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename=os.path.basename(url)
    expected_sha256=url.split("/")[-2]
    dst=os.path.join(root, filename)
    if os.path.isfile(dst):
        if hashlib.sha256(open(dst,"rb").read()).hexdigest()==expected_sha256: return dst
        warnings.warn(f"{dst} exists, but checksum mismatch; re-downloading")
    with urllib.request.urlopen(url) as src, open(dst,"wb") as out:
        with tqdm(total=int(src.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buf=src.read(8192)
                if not buf: break
                out.write(buf); loop.update(len(buf))
    if hashlib.sha256(open(dst,"rb").read()).hexdigest()!=expected_sha256:
        raise RuntimeError("Downloaded model checksum mismatch")
    return dst

def available_models(): return list(_MODELS.keys())

class LoRM(nn.Module):
    def __init__(self, dim, rank=8, dropout=0.0, max_frames=None):
        super().__init__()
        self.rank=rank
        self.dropout=nn.Dropout(dropout)
        self.W_down_s=nn.Linear(dim, rank, bias=False)
        self.W_up_s=nn.Linear(rank, dim, bias=False)
        self.W_down_b=nn.Linear(dim, rank, bias=False)
        self.W_up_b=nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.W_up_s.weight); nn.init.zeros_(self.W_up_b.weight)
        self.base_scale=nn.Parameter(torch.ones(dim))
        self.base_shift=nn.Parameter(torch.zeros(dim))
        self.max_frames=max_frames
        self.temporal_embed=nn.Parameter(torch.randn(max_frames, self.rank)*0.02) if max_frames is not None else None
    def forward(self, x, video_frame):
        if video_frame<=1: return x
        L,B,D=x.shape
        cls=x[0:1]; sp=x[1:]
        batch=B//video_frame
        sp=sp.permute(1,0,2).view(batch, video_frame, -1, D)
        if self.temporal_embed is None or self.max_frames is None:
            t=torch.zeros(video_frame, self.rank, device=x.device, dtype=x.dtype)
        else:
            t=self.temporal_embed
            if t.shape[0]!=video_frame:
                t=F.interpolate(t.unsqueeze(0).permute(0,2,1), size=video_frame, mode='linear', align_corners=False).permute(0,2,1).squeeze(0)
            t=t.to(dtype=x.dtype)
        down_scale=self.W_down_s(self.base_scale.to(self.W_down_s.weight.dtype))
        down_shift=self.W_down_b(self.base_shift.to(self.W_down_b.weight.dtype))
        z_scale=t*down_scale.unsqueeze(0); z_shift=t*down_shift.unsqueeze(0)
        scale=self.W_up_s(z_scale); shift=self.W_up_b(z_shift)
        scale=scale.unsqueeze(0).unsqueeze(2)+1.0; shift=shift.unsqueeze(0).unsqueeze(2)
        sp=sp*scale+shift; sp=self.dropout(sp)
        sp=sp.view(batch*video_frame, -1, D).permute(1,0,2).contiguous()
        return torch.cat([cls, sp], dim=0)

class AsyncMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, dropout=0.0, topk_ratio=0.5, max_off=2, topk_fixed=None):
        super().__init__()
        self.embed_dim=embed_dim; self.num_heads=num_heads; self.head_dim=embed_dim//num_heads
        self.scale=self.head_dim**-0.5
        self.topk_ratio=topk_ratio; self.topk_fixed=topk_fixed; self.max_off=max_off
        self.in_proj_weight=nn.Parameter(torch.empty(3*embed_dim, embed_dim))
        self.in_proj_bias=nn.Parameter(torch.empty(3*embed_dim)) if bias else None
        self.out_proj=nn.Linear(embed_dim, embed_dim, bias=bias)
        red=16
        self.gamma=nn.Sequential(nn.Linear(embed_dim, embed_dim//red, bias=False), nn.ReLU(), nn.Linear(embed_dim//red, embed_dim, bias=False))
        self.beta=nn.Sequential(nn.Linear(embed_dim, embed_dim//red, bias=False), nn.ReLU(), nn.Linear(embed_dim//red, embed_dim, bias=False))
        self.score_net=nn.Linear(embed_dim,1)
        self.off_param=nn.Parameter(torch.zeros(num_heads))
        self.attn_drop=nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None: nn.init.constant_(self.in_proj_bias,0.)
        nn.init.constant_(self.out_proj.bias,0.)
        nn.init.zeros_(self.gamma[-1].weight); nn.init.zeros_(self.beta[-1].weight)
    def _straight_through_topk(self, scores, k):
        if k<=0: return torch.zeros_like(scores)
        v,idx=torch.topk(scores,k,dim=1)
        hard=torch.zeros_like(scores).scatter_(1, idx, 1.0)
        return hard - scores.detach() + scores
    @staticmethod
    def _round_ste(x): return (x - x.detach()) + torch.round(x.detach())
    def forward(self, query, key, value, attn_mask=None, need_weights=False, video_frame=-1, text_global=None):
        L,B,D=query.shape
        w_in = self.in_proj_weight.to(query.dtype)
        b_in = self.in_proj_bias.to(query.dtype) if self.in_proj_bias is not None else None
        qkv=F.linear(query, w_in, b_in)
        q,k,v=qkv.chunk(3, dim=-1)
        q=q.view(L,B,self.num_heads,self.head_dim).permute(1,0,2,3)
        k=k.view(L,B,self.num_heads,self.head_dim).permute(1,0,2,3)
        v=v.view(L,B,self.num_heads,self.head_dim).permute(1,0,2,3)
        if video_frame>1:
            batch=B//video_frame
            scores=self.score_net(query.permute(1,0,2)).view(batch, video_frame, L)
            scores_sp=scores[:,:,1:]
            if text_global is not None:
                qperm=query.permute(1,0,2).view(batch, video_frame, L, D)
                q_sp=qperm[:,:,1:,:]
                sim=torch.einsum('btld,bd->btl', q_sp, text_global)
                scores_sp=scores_sp+sim
            if self.topk_fixed is not None:
                k_sel=min(int(self.topk_fixed), max(1, L-1))
            else:
                k_sel=min(int(self.topk_ratio), L-1) if float(self.topk_ratio)>=1 else max(1,int((L-1)*float(self.topk_ratio)))
            scores_sp_flat=scores_sp.contiguous().view(batch*video_frame, L-1)
            mask_sp_flat=self._straight_through_topk(scores_sp_flat, k_sel)
            ones=torch.ones((batch*video_frame,1), device=mask_sp_flat.device, dtype=mask_sp_flat.dtype)
            mask_all_flat=torch.cat([ones, mask_sp_flat], dim=1)
            mask_sel=mask_all_flat.view(batch*video_frame, L)
            t_global=query.mean(dim=0).view(batch, video_frame, -1).mean(dim=1)
            gam=self.gamma(t_global).view(batch,1,1,self.num_heads,self.head_dim)
            bet=self.beta(t_global).view(batch,1,1,self.num_heads,self.head_dim)
            q_r=q.view(batch, video_frame, L, self.num_heads, self.head_dim)
            q_r=q_r*(1.0+gam)+bet
            q=q_r.view(B,L,self.num_heads,self.head_dim)
            offsets=torch.clamp(self._round_ste(self.off_param), -self.max_off, self.max_off)
            k_r=k.view(batch, video_frame, L, self.num_heads, self.head_dim)
            v_r=v.view(batch, video_frame, L, self.num_heads, self.head_dim)
            ks,vs=[],[]
            for h in range(self.num_heads):
                s=int(offsets[h].item())
                ks.append(torch.roll(k_r[:,:,:,h], shifts=s, dims=1) if s!=0 else k_r[:,:,:,h])
                vs.append(torch.roll(v_r[:,:,:,h], shifts=s, dims=1) if s!=0 else v_r[:,:,:,h])
            k=torch.stack(ks, dim=3).view(B,L,self.num_heads,self.head_dim)
            v=torch.stack(vs, dim=3).view(B,L,self.num_heads,self.head_dim)
            m=mask_sel.to(dtype=q.dtype, device=q.device)
            mq=m.unsqueeze(1).unsqueeze(-1)
            mk=m.unsqueeze(1).unsqueeze(2)
            allowed= mq*mk
            attn_bias=(1.0-allowed)*torch.tensor(-1e4, dtype=q.dtype, device=q.device)
            attn_bias=attn_bias.expand(-1,self.num_heads,-1,-1)
            if attn_mask is None: attn_mask_combined=attn_bias
            else:
                bm=attn_mask
                if bm.dim()==2: bm=bm.unsqueeze(0).unsqueeze(0)
                elif bm.dim()==3: bm=bm.unsqueeze(1)
                bm=bm.to(dtype=attn_bias.dtype, device=attn_bias.device)
                attn_mask_combined=bm+attn_bias
        else:
            attn_mask_combined=attn_mask
        if isinstance(attn_mask_combined, torch.Tensor):
            neg_inf=torch.finfo(attn_mask_combined.dtype).min/2
            attn_mask_combined=torch.where(torch.isfinite(attn_mask_combined), attn_mask_combined, neg_inf)
            clamp_val=max(1e4, float(-neg_inf/2))
            attn_mask_combined=attn_mask_combined.clamp(min=-clamp_val, max=clamp_val)
        attn=F.scaled_dot_product_attention(q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3), attn_mask=attn_mask_combined, dropout_p=self.attn_drop.p if self.training else 0.0)
        attn=attn.permute(2,0,1,3).contiguous().view(L,B,self.embed_dim)
        out = F.linear(attn, self.out_proj.weight.to(attn.dtype), self.out_proj.bias.to(attn.dtype) if self.out_proj.bias is not None else None)
        return (out, None)


class Bottleneck(nn.Module):
    expansion=4
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(inplanes, planes, 1, bias=False); self.bn1=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes, planes, 3, padding=1, bias=False); self.bn2=nn.BatchNorm2d(planes)
        self.avgpool=nn.AvgPool2d(stride) if stride>1 else nn.Identity()
        self.conv3=nn.Conv2d(planes, planes*self.expansion, 1, bias=False); self.bn3=nn.BatchNorm2d(planes*self.expansion)
        self.relu=nn.ReLU(inplace=True); self.downsample=None; self.stride=stride
        if stride>1 or inplanes!=planes*Bottleneck.expansion:
            self.downsample=nn.Sequential(OrderedDict([("-1", nn.AvgPool2d(stride)), ("0", nn.Conv2d(inplanes, planes*self.expansion, 1, stride=1, bias=False)), ("1", nn.BatchNorm2d(planes*self.expansion))]))
    def forward(self, x):
        idt=x
        out=self.relu(self.bn1(self.conv1(x)))
        out=self.relu(self.bn2(self.conv2(out)))
        out=self.avgpool(out)
        out=self.bn3(self.conv3(out))
        if self.downsample is not None: idt=self.downsample(x)
        out+=idt; out=self.relu(out); return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim, embed_dim, num_heads, output_dim=None):
        super().__init__()
        self.positional_embedding=nn.Parameter(torch.randn(spacial_dim**2+1, embed_dim)/embed_dim**0.5)
        self.k_proj=nn.Linear(embed_dim, embed_dim)
        self.q_proj=nn.Linear(embed_dim, embed_dim)
        self.v_proj=nn.Linear(embed_dim, embed_dim)
        self.c_proj=nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads=num_heads
    def forward(self, x):
        x=x.reshape(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]).permute(2,0,1)
        x=torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x=x + self.positional_embedding[:, None, :].to(x.dtype)
        x,_=F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1], num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight, k_proj_weight=self.k_proj.weight, v_proj_weight=self.v_proj.weight,
            in_proj_weight=None, in_proj_bias=torch.cat([self.q_proj.bias,self.k_proj.bias,self.v_proj.bias]),
            bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0, out_proj_weight=self.c_proj.weight, out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True, training=self.training, need_weights=False
        )
        return x[0]

class ModifiedResNet(nn.Module):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim=output_dim; self.input_resolution=input_resolution
        self.conv1=nn.Conv2d(3, width//2, 3, stride=2, padding=1, bias=False); self.bn1=nn.BatchNorm2d(width//2)
        self.conv2=nn.Conv2d(width//2, width//2, 3, padding=1, bias=False); self.bn2=nn.BatchNorm2d(width//2)
        self.conv3=nn.Conv2d(width//2, width, 3, padding=1, bias=False); self.bn3=nn.BatchNorm2d(width)
        self.avgpool=nn.AvgPool2d(2); self.relu=nn.ReLU(inplace=True); self._inplanes=width
        self.layer1=self._make_layer(width, layers[0]); self.layer2=self._make_layer(width*2, layers[1], stride=2)
        self.layer3=self._make_layer(width*4, layers[2], stride=2); self.layer4=self._make_layer(width*8, layers[3], stride=2)
        embed_dim=width*32; self.attnpool=AttentionPool2d(input_resolution//32, embed_dim, heads, output_dim)
    def _make_layer(self, planes, blocks, stride=1):
        layers=[Bottleneck(self._inplanes, planes, stride)]; self._inplanes=planes*Bottleneck.expansion
        for _ in range(1, blocks): layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1,self.bn1),(self.conv2,self.bn2),(self.conv3,self.bn3)]:
                x=self.relu(bn(conv(x)))
            return self.avgpool(x)
        x=x.type(self.conv1.weight.dtype)
        x=stem(x); x=self.layer1(x); x=self.layer2(x); x=self.layer3(x); x=self.layer4(x)
        return self.attnpool(x)

class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        t=x.dtype
        y=super().forward(x.type(torch.float32))
        return y.type(t)

class QuickGELU(nn.Module):
    def forward(self, x): return x*torch.sigmoid(1.702*x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None, adapter_config:dict=None):
        super().__init__()
        self.use_asa=adapter_config.get('use_asa', False) if adapter_config else False
        self.use_lorm=adapter_config.get('use_lorm', False) if adapter_config else False
        self.attn=AsyncMultiheadAttention(d_model, n_head,
                                          topk_ratio=adapter_config.get('asa_topk',0.5),
                                          max_off=adapter_config.get('asa_max_off',2),
                                          topk_fixed=adapter_config.get('asa_topk_fixed',None)) if self.use_asa else nn.MultiheadAttention(d_model, n_head)
        if self.use_lorm:
            self.lorm=LoRM(d_model, rank=adapter_config.get('lorm_rank',8), dropout=adapter_config.get('lorm_dropout',0.0), max_frames=adapter_config.get('max_frames',None))
        self.ln_1=LayerNorm(d_model)
        self.mlp=nn.Sequential(OrderedDict([("c_fc", nn.Linear(d_model, d_model*4)), ("gelu", QuickGELU()), ("c_proj", nn.Linear(d_model*4, d_model))]))
        self.ln_2=LayerNorm(d_model)
        self.attn_mask=attn_mask
    def attention(self, x, video_frame=-1, text_global=None):
        m=self.attn_mask
        if m is not None and hasattr(m,'__call__'): m=m(x.size(0))
        m=m.to(dtype=x.dtype, device=x.device) if m is not None else None
        if isinstance(self.attn, AsyncMultiheadAttention): return self.attn(x,x,x, attn_mask=m, video_frame=video_frame, text_global=text_global)[0]
        else: return self.attn(x,x,x, need_weights=False, attn_mask=m)[0]
    def forward(self, x_tuple):
        if len(x_tuple)==3: x,video_frame,text_global=x_tuple
        else: x,video_frame=x_tuple; text_global=None
        xa=self.ln_1(x); 
        if self.use_lorm: xa=self.lorm(xa, video_frame)
        x = x + self.attention(xa, video_frame=video_frame, text_global=text_global)
        xm=self.ln_2(x)
        if self.use_lorm: xm=self.lorm(xm, video_frame)
        x = x + self.mlp(xm)
        return (x,video_frame,text_global) if text_global is not None else (x,video_frame)

class Transformer(nn.Module):
    def __init__(self, width, layers, heads, attn_mask=None, adapter_config:dict=None):
        super().__init__()
        self.width=width; self.layers=layers
        self.resblocks=nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, adapter_config) for _ in range(layers)])
    def forward(self, x, video_frame=-1, text_global=None):
        return self.resblocks((x,video_frame,text_global))[0]

class VisualTransformer(nn.Module):
    def __init__(self, input_resolution, patch_size, width, layers, heads, output_dim, linear_patch='2d', adapter_config:dict=None):
        super().__init__()
        self.input_resolution=input_resolution; self.output_dim=output_dim
        self.conv1=nn.Conv2d(3, width, patch_size, stride=patch_size, bias=False)
        scale=width**-0.5
        self.class_embedding=nn.Parameter(scale*torch.randn(width))
        self.positional_embedding=nn.Parameter(scale*torch.randn((input_resolution//patch_size)**2+1, width))
        self.ln_pre=LayerNorm(width)
        self.transformer=Transformer(width, layers, heads, adapter_config=adapter_config)
        self.ln_post=LayerNorm(width)
        self.proj=nn.Parameter(scale*torch.randn(width, output_dim))
        assert linear_patch in ['2d','3d']
        self.linear_patch=linear_patch
        if self.linear_patch=='3d':
            self.conv2=nn.Conv3d(3, width, kernel_size=(3,patch_size,patch_size), stride=(1,patch_size,patch_size), padding=(1,0,0), bias=False)
    def forward(self, x, video_frame=-1, text_global=None):
        if self.linear_patch=='3d':
            assert video_frame!=-1
            x3=x.reshape(-1, video_frame, x.shape[-3], x.shape[-2], x.shape[-1]).permute(0,2,1,3,4)
            x=self.conv2(x3).permute(0,2,1,3,4).reshape(-1, x3.shape[-3], x3.shape[-2], x3.shape[-1]).contiguous()
        else:
            x=self.conv1(x)
        x=x.reshape(x.shape[0], x.shape[1], -1).permute(0,2,1)
        x=torch.cat([self.class_embedding.to(x.dtype)+torch.zeros(x.shape[0],1,x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x=x + self.positional_embedding.to(x.dtype)
        x=self.ln_pre(x); x=x.permute(1,0,2)
        x=self.transformer(x, video_frame=video_frame, text_global=text_global)
        x=x.permute(1,0,2)
        return x

class CLIP(nn.Module):
    def __init__(self, embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size, context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, linear_patch='2d', adapter_config:dict=None):
        super().__init__()
        self.context_length=context_length
        if adapter_config is not None: vcfg=dict(adapter_config); tcfg={'use_asa':False,'use_lorm':False}
        else: vcfg=None; tcfg=None
        if isinstance(vision_layers,(tuple,list)):
            vision_heads=vision_width*32//64
            self.visual=ModifiedResNet(vision_layers, embed_dim, vision_heads, image_resolution, vision_width)
        else:
            vision_heads=vision_width//64
            self.visual=VisualTransformer(image_resolution, vision_patch_size, vision_width, vision_layers, vision_heads, embed_dim, linear_patch, vcfg)
        self.transformer=Transformer(transformer_width, transformer_layers, transformer_heads, attn_mask=self.build_attention_mask, adapter_config=tcfg)
        self.vocab_size=vocab_size
        self.token_embedding=nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding=nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final=LayerNorm(transformer_width)
        self.text_projection=nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale=nn.Parameter(torch.ones([]))
        self.text2vis_proj=nn.Linear(transformer_width, vision_width, bias=True)
        self.initialize_parameters()
    
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std=self.visual.attnpool.c_proj.in_features**-0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
            for block in [self.visual.layer1,self.visual.layer2,self.visual.layer3,self.visual.layer4]:
                for name,p in block.named_parameters():
                    if name.endswith("bn3.weight"): nn.init.zeros_(p)
        proj_std=(self.transformer.width**-0.5)*((2*self.transformer.layers)**-0.5)
        attn_std=self.transformer.width**-0.5
        fc_std=(2*self.transformer.width)**-0.5
        for block in self.transformer.resblocks:
            if hasattr(block.attn,'in_proj_weight'):
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None: nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)
        nn.init.normal_(self.text2vis_proj.weight, std=self.transformer.width**-0.5)
        nn.init.constant_(self.text2vis_proj.bias, 0.0)
    
    @staticmethod
    def get_config(pretrained_clip_name="ViT-B/32"):
        model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViT-B-32.pt")
        if pretrained_clip_name in _MODELS and pretrained_clip_name in _PT_NAME:
            model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[pretrained_clip_name])
        if pretrained_clip_name in ["ViT-B/32","ViT-B/16"] and os.path.exists(model_path):
            pass
        else:
            if pretrained_clip_name in _MODELS: model_path=_download(_MODELS[pretrained_clip_name])
            elif os.path.isfile(pretrained_clip_name): model_path=pretrained_clip_name
            else: raise RuntimeError(f"Model {pretrained_clip_name} not found")
        try:
            model=torch.jit.load(model_path, map_location="cpu").eval(); state_dict=model.state_dict()
        except RuntimeError:
            state_dict=torch.load(model_path, map_location="cpu")
        return state_dict
    
    def build_attention_mask(self, context_length):
        m=torch.zeros(context_length, context_length); m.fill_(float("-inf")); m.triu_(1); return m
    
    @property
    def dtype(self): return self.visual.conv1.weight.dtype
    def encode_image(self, image, return_hidden=False, video_frame=-1, text_global=None):
        tg=None
        if text_global is not None:
            tg = F.linear(text_global.to(dtype=self.dtype), self.text2vis_proj.weight.to(dtype=self.dtype), self.text2vis_proj.bias.to(dtype=self.dtype))
        hidden=self.visual(image.type(self.dtype), video_frame=video_frame, text_global=tg)
        hidden=self.visual.ln_post(hidden) @ self.visual.proj
        x=hidden[:,0,:]
        if return_hidden: return x, hidden
        return x
    
    def encode_text(self, text, return_hidden=False):
        x = self.token_embedding(text)
        t_dtype = next(self.transformer.parameters()).dtype
        x = x.to(t_dtype)
        pos = self.positional_embedding[:x.size(1), :].to(t_dtype)
        x = x + pos
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        h = self.ln_final(x).to(self.text_projection.dtype)
        hidden = h @ self.text_projection
        idx = text.argmax(dim=-1)
        x = hidden[torch.arange(hidden.shape[0]), idx]
        if return_hidden:
            return x, hidden
        return x


    def encode_text_global(self, text):
        x=self.token_embedding(text).type(self.dtype)
        pos=self.positional_embedding[:x.size(1),:].type(self.dtype)
        x=x+pos; x=x.permute(1,0,2); x=self.transformer(x); x=x.permute(1,0,2)
        hidden=self.ln_final(x).type(self.dtype)
        idx=text.argmax(dim=-1)
        return hidden[torch.arange(hidden.shape[0]), idx]
    
    def forward(self, image, text):
        im=self.encode_image(image); tx=self.encode_text(text)
        im=im/im.norm(dim=-1, keepdim=True).clamp_min(1e-6); tx=tx/tx.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        s=self.logit_scale.exp()
        return s*im@tx.t(), s*tx@im.t()

def convert_weights(model: nn.Module):
    def _convert(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None: l.bias.data = l.bias.data.half()
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in","q","k","v"]], "in_proj_bias","bias_k","bias_v"]:
                if hasattr(l, attr):
                    t=getattr(l, attr)
                    if t is not None: t.data=t.data.half()
        for name in ["text_projection","proj"]:
            if hasattr(l, name):
                a=getattr(l, name)
                if isinstance(a, nn.Parameter): a.data=a.data.half()
                elif hasattr(a,"weight") and hasattr(a,"bias"):
                    a.weight.data=a.weight.data.half()
                    if a.bias is not None: a.bias.data=a.bias.data.half()
    def _skip_custom(l):
        if isinstance(l, AsyncMultiheadAttention): return True
        if isinstance(l, LoRM): return True
        if hasattr(l, "text2vis_proj"): return True
        return False
    for m in model.modules():
        if _skip_custom(m): continue
        _convert(m)

def build_model(state_dict: dict):
    vit="visual.proj" in state_dict
    if vit:
        vision_width=state_dict["visual.conv1.weight"].shape[0]
        vision_layers=len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size=state_dict["visual.conv1.weight"].shape[-1]
        grid=round((state_dict["visual.positional_embedding"].shape[0]-1)**0.5)
        image_resolution=vision_patch_size*grid
    else:
        counts=[len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1,2,3,4]]
        vision_layers=tuple(counts)
        vision_width=state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width=round((state_dict["visual.attnpool.positional_embedding"].shape[0]-1)**0.5)
        vision_patch_size=None
        assert output_width**2+1==state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution=output_width*32
    embed_dim=state_dict["text_projection"].shape[1]
    context_length=state_dict["positional_embedding"].shape[0]
    vocab_size=state_dict["token_embedding.weight"].shape[0]
    transformer_width=state_dict["ln_final.weight"].shape[0]
    transformer_heads=transformer_width//64
    transformer_layers=len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    model=CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size, context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)
    for key in ["input_resolution","context_length","vocab_size"]:
        if key in state_dict: del state_dict[key]
    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()