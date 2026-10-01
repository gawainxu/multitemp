import torch
from torch import nn
from torch.nn import Module, ModuleList

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import ml_collections

from dataUtil import image_size_mapping


# helpers



def get_b16_config_cifar():
    """
    Returns the ViT-B/16 configuration.
    https://arxiv.org/pdf/2203.08441 for ImageNet
    https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/11-vision-transformer.html for Cifar
    """
    config = ml_collections.ConfigDict()
    config.patch_size = 4
    config.embed_dim = 256    # output dim for converting patches to attention inputs, 256
    config.emb_dropout = 0.2
    config.hidden_dim = 512   # in FFN in transformer, 512
    config.depth = 6
    config.attention_dropout = 0.3
    config.head_dim = 64
    config.num_heads = 8
    config.dropout = 0.3

    return config


def get_b16_config():
    """
    Returns the ViT-B/16 configuration.
    https://arxiv.org/pdf/2203.08441 for ImageNet
    https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/11-vision-transformer.html for Cifar
    """
    config = ml_collections.ConfigDict()
    config.patch_size = 16
    config.embed_dim = 768
    config.emb_dropout = 0.1
    config.hidden_dim = 3072
    config.depth = 12
    config.attention_dropout = 0.2
    config.head_dim = 64
    config.num_heads = 12
    config.dropout = 0.1

    return config



def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(Module):
    def __init__(self, embedding_dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(self, embedding_dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == embedding_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(embedding_dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(embedding_dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, embedding_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(Module):
    def __init__(self, embedding_dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(embedding_dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(embedding_dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)



class ViT(Module):

    # according to https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf
    # https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/11-vision-transformer.html
    def __init__(self, *, image_size, patch_size, num_classes, embedding_dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        # the image is square by default
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_cls_tokens = 1 if pool == 'cls' else 0

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.cls_token = nn.Parameter(torch.randn(num_cls_tokens, embedding_dim))
        self.pos_embedding = nn.Parameter(torch.randn(num_patches + num_cls_tokens, embedding_dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(embedding_dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(embedding_dim, num_classes) if num_classes > 0 else None

    def forward(self, img):
        batch = img.shape[0]
        x = self.to_patch_embedding(img)

        cls_tokens = repeat(self.cls_token, '... d -> b ... d', b = batch)
        x = torch.cat((cls_tokens, x), dim = 1)

        seq = x.shape[1]

        x = x + self.pos_embedding[:seq]
        x = self.dropout(x)

        x = self.transformer(x)

        if self.mlp_head is None:
            return x

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


if __name__ == "__main__":
    configs = get_b16_config()
    print(configs)
    image_size = 224
    patch_size = 4
    vit = ViT(image_size=image_size, patch_size=patch_size, num_classes=50, dim=64, depth=16, heads=8, mlp_dim=configs.hidden_size)