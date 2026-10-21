"""Sprint17 candidate v2: torchvision MViT-B (Kinetics-400 layout) + DETR-style decoder.  v1 (model_mvit_detr.py) is kept unchanged.
Changes vs v1, each from the code review:
  1. explicit per-view learned 3-D positional embeddings (t,h,w + class token) are added to the decoder memory;
  2. the decoder queries start at N(0,1) scale (v1 used std 0.02, which makes the slots almost identical at the start);
  3. the token layout is asserted at run time: 1 class token + 8*7*7 spatio-temporal tokens per view, 768-d;
  4. strict loading of pretrained weights reports missing/unexpected keys by name and raises on any difference.
The classification path of torchvision (`x = x[:, 0]`; `self.head`) is never used: `tokens()` re-implements MViT.forward up to the final LayerNorm and returns every token."""
import math
import torch
from torch import nn

FINAL_GRID = (8, 7, 7)          # (T', H', W') after the last MultiscaleBlock for T=16, 224x224 (checked at run time)
N_TOK = 1 + math.prod(FINAL_GRID)

class MViTDETRv2(nn.Module):
    def __init__(self, cfg, d_model=256, nheads=8, dec_layers=3, dim_ff=1024, dropout=0.1, query_std=1.0, backbone_state=None):
        super().__init__()
        from torchvision.models.video import mvit_v1_b
        self.slots, self.gap = cfg['slots'], cfg['minimum_predicted_interval']
        bb = mvit_v1_b(weights=None)
        self.load_report = None
        if backbone_state is not None:                                   # load into THIS object, with its 400-class head still present
            res = bb.load_state_dict(backbone_state, strict=False)
            self.load_report = {'missing': list(res.missing_keys), 'unexpected': list(res.unexpected_keys), 'tensors_in_file': len(backbone_state)}
            if res.missing_keys or res.unexpected_keys: raise RuntimeError('BACKBONE_STATE_MISMATCH:' + repr(self.load_report))
        bb.head = nn.Identity()
        self.backbone = bb
        c = bb.norm.normalized_shape[0]
        self.proj = nn.Linear(c, d_model)
        t, h, w = FINAL_GRID
        self.pos_t = nn.Parameter(torch.randn(2, t, d_model) * 0.02); self.pos_h = nn.Parameter(torch.randn(2, h, d_model) * 0.02)
        self.pos_w = nn.Parameter(torch.randn(2, w, d_model) * 0.02); self.pos_cls = nn.Parameter(torch.randn(2, 1, d_model) * 0.02)
        self.view_embed = nn.Parameter(torch.randn(2, 1, d_model) * 0.02)
        self.queries = nn.Parameter(torch.randn(self.slots, d_model) * query_std)
        layer = nn.TransformerDecoderLayer(d_model, nheads, dim_ff, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, dec_layers, norm=nn.LayerNorm(d_model))
        self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 7))
        nn.init.normal_(self.out[-1].weight, std=0.01); nn.init.zeros_(self.out[-1].bias)

    def tokens(self, x):
        if x.ndim != 5 or tuple(x.shape[1:]) != (3, 16, 224, 224): raise ValueError('INPUT_MUST_BE_B_3_16_224_224_GOT_' + str(tuple(x.shape)))
        bb = self.backbone
        x = bb.conv_proj(x).flatten(2).transpose(1, 2)
        x = bb.pos_encoding(x)
        thw = (bb.pos_encoding.temporal_size,) + bb.pos_encoding.spatial_size
        for block in bb.blocks: x, thw = block(x, thw)
        x = bb.norm(x)
        if tuple(thw) != FINAL_GRID or x.shape[1] != N_TOK: raise AssertionError('TOKEN_LAYOUT_CHANGED_%s_%s' % (tuple(thw), tuple(x.shape)))
        return x

    def position(self, i):
        t, h, w = FINAL_GRID
        p = (self.pos_t[i][:, None, None] + self.pos_h[i][None, :, None] + self.pos_w[i][None, None, :]).reshape(t * h * w, -1)
        return torch.cat([self.pos_cls[i], p], 0)                       # [1+392, d]; row 0 = class token, then t-major, then h, then w

    def forward(self, views):
        if set(views) != {'rank0', 'rank1'}: raise ValueError('BOTH_VIEWS_REQUIRED')
        mem = [self.proj(self.tokens(views[rk])) + self.position(i)[None] + self.view_embed[i] for i, rk in enumerate(('rank0', 'rank1'))]
        mem = torch.cat(mem, 1); B = mem.shape[0]
        h = self.decoder(self.queries[None].expand(B, -1, -1), mem)
        raw = self.out(h).float()
        start = (1 - self.gap) * raw[..., 1].sigmoid()
        end = start + self.gap + (1 - self.gap - start) * raw[..., 2].sigmoid()
        return {'confidence_logits': raw[..., 0], 'intervals': torch.stack([start, end], -1), 'points': raw[..., 3:7].sigmoid().reshape(B, self.slots, 2, 2)}

def param_groups(model, lr_backbone, lr_head, weight_decay=0.05):
    bb = [p for p in model.backbone.parameters() if p.requires_grad]
    hd = [p for n, p in model.named_parameters() if not n.startswith('backbone.') and p.requires_grad]
    return [{'params': bb, 'lr': lr_backbone, 'weight_decay': weight_decay}, {'params': hd, 'lr': lr_head, 'weight_decay': weight_decay}]
