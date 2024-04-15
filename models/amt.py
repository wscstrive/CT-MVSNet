import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import to_2tuple

class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_models, max_shape=(700, 700), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/4 featmap, the max length of 600 corresponds to 2400 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()
        self.max_shape = max_shape
        self.temp_bug_fix = temp_bug_fix
        self.d_model = d_models
        pe = torch.zeros((self.d_model, *self.max_shape))
        y_position = torch.ones(self.max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(self.max_shape).cumsum(1).float().unsqueeze(0)
        if self.temp_bug_fix:
            div_term = torch.exp(torch.arange(0, self.d_model//2, 2).float() * (-math.log(10000.0) / (self.d_model//2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, self.d_model//2, 2).float() * (-math.log(10000.0) / self.d_model//2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]
        # self.sigmpod = nn.Sigmoid()
        # self.register_buffer('pe11', pe.unsqueeze(0))  # [1, C, H, W]
    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """

        return x + self.pe[:, :, :x.size(2), :x.size(3)]
        # return x * self.sigmoid(self.pe[:, :, :x.size(2), :x.size(3)])

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.pos = PositionEncodingSine(embed_dim)

    def forward(self, x):

        _,_,H,W = x.shape
        x = self.proj(x)
        x = self.pos(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, H, W

class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super(LinearAttention, self).__init__()
        self.feature_map = lambda x: torch.nn.functional.elu(x) + 1
        self.eps = eps

    def forward(self, queries, keys, values):
        """Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
        Returns:
            V: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)


        v_length = values.size(1)
        values = values / v_length
        KV = torch.einsum("nshd,nshm->nhmd", K, values)  # (2,8,4,4)

        # Compute the normalizer
        Z = 1/(torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))+self.eps)

        # Finally compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z) * v_length

        return V.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        # Fill d_keys and d_values
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        values = self.value_projection(values).view(N, S, H, -1)

        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
        ).view(N, L, -1)

        # Project the output and return
        return self.out_projection(new_values)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, d_ff=None, dropout=0.0,
                 activation="relu"):
        super(EncoderLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        inner_attention = LinearAttention()
        attention = AttentionLayer(inner_attention, d_model, n_heads, d_keys, d_values)

        d_ff = d_ff or 2 * d_model
        self.attention = attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x, source):
        # Run self attention and add it to the input
        x = x + self.dropout(self.attention(x, source, source))

        # MLP-like
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y)

class AMT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, d_models=32, n_heads=8,
                 layers_name=['self','self','cross'],embed_dims=32):
        super(AMT, self).__init__()
        self.d_model = d_models
        self.nhead = n_heads
        self.layers_name = layers_name
        self.embed_dim = embed_dims
        self.PatchEmbed = PatchEmbed(img_size=img_size,
                                     patch_size=patch_size,
                                     in_chans=d_models,
                                     embed_dim=embed_dims)
        encoder_layer = EncoderLayer(embed_dims, n_heads)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layers_name))])
        self._reset_parameters()

        # self.pos_encoding = PositionEncodingSuperGule(config['d_model'])
        self.pos_encoding = PositionEncodingSine(d_models)


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, ref_feature=None, src_feature=None, feat="ref"):
        """
        Args:
            ref_feature(torch.Tensor): [N, C, H, W]
            src_feature(torch.Tensor): [N, C, H, W]
        """

        assert ref_feature is not None

        if feat == "ref": # only self attention layer

            assert self.d_model == ref_feature.size(1)
            ref_feature, H, W = self.PatchEmbed(ref_feature)
            ref_feature_list = []
            for layer, name in zip(self.layers, self.layers_name): # every self attention layer
                if name == 'self':
                    ref_feature = layer(ref_feature, ref_feature)
                    ref_feature_list.append(einops.rearrange(ref_feature, 'n (h w) c -> n c h w', h=H))

            return ref_feature_list

        elif feat == "src":

            assert self.embed_dim == ref_feature[0].size(1)
            src_feature, H, W = self.PatchEmbed(src_feature)

            ref_feature = [einops.rearrange(_, 'n c h w -> n (h w) c') for _ in ref_feature]

            for i, (layer, name) in enumerate(zip(self.layers, self.layers_name)):
                if name == 'self':
                    src_feature = layer(src_feature, src_feature)
                elif name == 'cross':
                    src_feature = layer(src_feature, ref_feature[0])
                else:
                    raise KeyError
            return einops.rearrange(src_feature, 'n (h w) c -> n c h w', h=H)
        else:
            raise ValueError("Wrong feature name")

class AdapMatchawareTransformer(nn.Module):
    def __init__(self,base_channels=8, img_size=224, patch_size=[1,2,4], d_models=[32, 16, 8], n_heads=[8, 8, 8],
                 layers_name=[['self', 'cross', 'cross'], ['self', 'cross'], ['self', 'self', 'cross']],
                 embed_dims=[32,32,32]):

        super(AdapMatchawareTransformer, self).__init__()


        self.AMT1 = AMT(img_size, patch_size[0], d_models[0], n_heads[0], layers_name[0], embed_dims[0]) # (b,c,h,w)
        self.AMT2 = AMT(img_size, patch_size[1], d_models[1], n_heads[1], layers_name[1], embed_dims[1]) # (b,c,h/2,w/2)
        self.AMT3 = AMT(img_size, patch_size[2], d_models[2], n_heads[2], layers_name[2], embed_dims[2]) # (b,c,h/4,w/4)

        self.dim_reduction_1 = nn.Conv2d(base_channels * 4, base_channels * 2, 1, bias=False)
        self.dim_reduction_2 = nn.Conv2d(base_channels * 4, base_channels * 1, 1, bias=False)

        self.smooth_1 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1, bias=False)
        self.smooth_2 = nn.Conv2d(base_channels * 1, base_channels * 1, 3, padding=1, bias=False)

    def _upsample_add(self, x, y):
        """_upsample_add. Upsample and add two feature maps.

        :param x: top feature map to be upsampled.
        :param y: lateral feature map.
        """

        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y


    def forward(self, features):
        """forward.

        :param features: multiple views and multiple stages features
        """

        for nview_idx, feature_multi_stages in enumerate(features):
            if nview_idx == 0: # ref view
                ref_fea_t_list_1 = self.AMT1(feature_multi_stages["stage1"].clone(), feat="ref")
                feature_multi_stages["stage1"] = ref_fea_t_list_1[-1]

                ref_fea_t_list_2 = self.AMT2(feature_multi_stages["stage2"].clone(), feat="ref")
                feature_multi_stages["stage2"] = self.smooth_1(self._upsample_add(self.dim_reduction_1(ref_fea_t_list_2[-1]),
                                                                                  feature_multi_stages["stage2"]))

                ref_fea_t_list_3 = self.AMT3(feature_multi_stages["stage3"].clone(), feat="ref")
                feature_multi_stages["stage3"] = self.smooth_2(self._upsample_add(self.dim_reduction_2(ref_fea_t_list_3[-1]),
                                                                                  feature_multi_stages["stage3"]))

            else: # src view
                feature_multi_stages["stage1"] = self.AMT1([_.clone() for _ in ref_fea_t_list_1], feature_multi_stages["stage1"].clone(),
                                                           feat="src")

                feature_multi_stages["stage2"] = self.smooth_1(self._upsample_add(self.dim_reduction_1(
                                                 self.AMT2([_.clone() for _ in ref_fea_t_list_2], feature_multi_stages["stage2"].clone(),
                                                           feat="src")), feature_multi_stages["stage2"]))

                feature_multi_stages["stage3"] = self.smooth_2(self._upsample_add(self.dim_reduction_2(
                                                 self.AMT3([_.clone() for _ in ref_fea_t_list_3], feature_multi_stages["stage3"].clone(),
                                                           feat="src")),feature_multi_stages["stage3"]))

        return features