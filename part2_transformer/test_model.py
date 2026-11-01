"""Sanity checks for YourModel. Run: python test_model.py"""
import torch
import torch.nn.functional as F

from main import YourModel, _Block
from models import VanillaEEGTransformer

torch.manual_seed(0)
x = torch.randn(4, 64, 481)

m = YourModel(64, 481).eval()
logits, emb = m(x, return_embed=True, keep=True)
assert logits.shape == (4, 2) and emb.shape == (4, 64)
assert m.n_tokens == 15                                   # 481 // 32 -> 15 patches (+CLS = 16 tokens)
a = m.blocks[0].last_attn
assert a.shape == (4, 4, 16, 16)                          # (batch, heads, tokens, tokens)
assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5)   # softmax rows sum to 1

# Hand-written scaled dot-product attention == PyTorch reference (scale 1/sqrt(d_k)).
blk = _Block(64, 4, 0.0, 0.0).eval()
z = torch.randn(2, 16, 64)
q, k, v = blk.qkv(blk.ln1(z)).view(2, 16, 3, 4, 16).permute(2, 0, 3, 1, 4)
ref = F.scaled_dot_product_attention(q, k, v)
mine = ((q @ k.transpose(-2, -1) / 16 ** 0.5).softmax(-1)) @ v
assert torch.allclose(ref, mine, atol=1e-5)

# Vanilla baseline is the untouched starter: 481 + CLS tokens.
assert VanillaEEGTransformer(64, 481).pos.shape[1] == 482
print("OK: shapes, softmax rows, attention scaling vs PyTorch reference, vanilla untouched")
