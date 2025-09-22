import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import warnings
import torch.utils.checkpoint as cp

try:
    from mmengine.model.weight_init import (constant_init, normal_init, trunc_normal_init)
    from mmengine.model import BaseModule, ModuleList, Sequential
    from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
    from mmcv.cnn.bricks.drop import build_dropout
    from mmcv.cnn.bricks.transformer import MultiheadAttention as MMCVMultiheadAttention
    MMSeg_AVAILABLE = True
except ImportError:
    try:
        from mmcv.cnn.utils.weight_init import (constant_init, normal_init, trunc_normal_init)
        from mmcv.runner import BaseModule, ModuleList, Sequential
        from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
        from mmcv.cnn.bricks.drop import build_dropout
        from mmcv.cnn.bricks.transformer import MultiheadAttention as MMCVMultiheadAttention
        MMSeg_AVAILABLE = True
    except ImportError as e_mmcv:
        MixVisionTransformer = None
        MMSeg_AVAILABLE = False

if MMSeg_AVAILABLE:

    def nchw_to_nlc(x):
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2)

    def nlc_to_nchw(x, hw_shape):
        B, N, C = x.shape
        H, W = hw_shape
        return x.transpose(1, 2).reshape(B, C, H, W)

    class _PatchEmbed(BaseModule):
        def __init__(self,
                     in_channels=3,
                     embed_dims=768,
                     kernel_size=16,
                     stride=16,
                     padding=0,
                     dilation=1,
                     norm_cfg=None,
                     init_cfg=None):
            super(_PatchEmbed, self).__init__(init_cfg)
            if stride is None:
                stride = kernel_size
            self.projection = Conv2d(
                in_channels,
                embed_dims,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation)
            if norm_cfg is not None:
                self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
            else:
                self.norm = None

        def forward(self, x):
            x = self.projection(x)
            hw_shape = (x.shape[2], x.shape[3])
            x = nchw_to_nlc(x)
            if self.norm is not None:
                x = self.norm(x)
            return x, hw_shape

    class MixFFN(BaseModule):
        def __init__(self,
                     embed_dims,
                     feedforward_channels,
                     act_cfg=dict(type='GELU'),
                     ffn_drop=0.,
                     dropout_layer=None,
                     init_cfg=None):
            super(MixFFN, self).__init__(init_cfg=init_cfg)
            self.embed_dims = embed_dims
            self.feedforward_channels = feedforward_channels
            self.act_cfg = act_cfg
            self.activate = build_activation_layer(act_cfg)
            in_channels = embed_dims
            fc1 = Conv2d(
                in_channels=in_channels,
                out_channels=feedforward_channels,
                kernel_size=1, stride=1, bias=True)
            pe_conv = Conv2d(
                in_channels=feedforward_channels,
                out_channels=feedforward_channels,
                kernel_size=3, stride=1, padding=(3 - 1) // 2, bias=True,
                groups=feedforward_channels)
            fc2 = Conv2d(
                in_channels=feedforward_channels,
                out_channels=in_channels,
                kernel_size=1, stride=1, bias=True)
            drop = nn.Dropout(ffn_drop)
            layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
            self.layers = Sequential(*layers)
            self.dropout_layer = build_dropout(
                dropout_layer) if dropout_layer else torch.nn.Identity()

        def forward(self, x, hw_shape, identity=None):
            out = nlc_to_nchw(x, hw_shape)
            out = self.layers(out)
            out = nchw_to_nlc(out)
            if identity is None: identity = x
            return identity + self.dropout_layer(out)

    class _EfficientMultiheadAttention(BaseModule):
        def __init__(self,
                     embed_dims,
                     num_heads,
                     attn_drop=0.,
                     proj_drop=0.,
                     dropout_layer=None,
                     batch_first=True,
                     qkv_bias=True,
                     norm_cfg=dict(type='LN'),
                     sr_ratio=1,
                     init_cfg=None):
            super().__init__(init_cfg=init_cfg)
            self.embed_dims = embed_dims
            self.num_heads = num_heads
            self.sr_ratio = sr_ratio
            self.attn = MMCVMultiheadAttention(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                dropout_layer={'type': 'Dropout', 'drop_prob': proj_drop},
                batch_first=batch_first,
                bias=qkv_bias)
            if sr_ratio > 1:
                self.sr = Conv2d(
                    in_channels=embed_dims,
                    out_channels=embed_dims,
                    kernel_size=sr_ratio,
                    stride=sr_ratio)
                self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
            self.dropout_layer = build_dropout(
                dropout_layer) if dropout_layer else torch.nn.Identity()

        def forward(self, x, hw_shape, identity=None):
            x_q = x
            if self.sr_ratio > 1:
                x_kv = nlc_to_nchw(x, hw_shape)
                x_kv = self.sr(x_kv)
                x_kv = nchw_to_nlc(x_kv)
                x_kv = self.norm(x_kv)
            else:
                x_kv = x
            if identity is None:
                identity = x_q
            out = self.attn(query=x_q, key=x_kv, value=x_kv)
            return identity + self.dropout_layer(out)

    class TransformerEncoderLayer(BaseModule):
        def __init__(self,
                     embed_dims,
                     num_heads,
                     feedforward_channels,
                     drop_rate=0.,
                     attn_drop_rate=0.,
                     drop_path_rate=0.,
                     qkv_bias=True,
                     act_cfg=dict(type='GELU'),
                     norm_cfg=dict(type='LN'),
                     batch_first=True,
                     sr_ratio=1,
                     with_cp=False):
            super(TransformerEncoderLayer, self).__init__()
            self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
            self.attn = _EfficientMultiheadAttention(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                batch_first=batch_first,
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg,
                sr_ratio=sr_ratio)
            self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
            self.ffn = MixFFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                act_cfg=act_cfg)
            self.with_cp = with_cp

        def forward(self, x, hw_shape):
            def _inner_forward(x):
                x = self.attn(self.norm1(x), hw_shape, identity=x)
                x = self.ffn(self.norm2(x), hw_shape, identity=x)
                return x
            if self.with_cp and x.requires_grad:
                x = cp.checkpoint(_inner_forward, x)
            else:
                x = _inner_forward(x)
            return x

    class MixVisionTransformer(BaseModule):
        def __init__(self,
                     in_channels=3,
                     embed_dims=64,
                     num_stages=4,
                     num_layers=[3, 4, 6, 3],
                     num_heads=[1, 2, 4, 8],
                     patch_sizes=[7, 3, 3, 3],
                     strides=[4, 2, 2, 2],
                     sr_ratios=[8, 4, 2, 1],
                     out_indices=(0, 1, 2, 3),
                     mlp_ratio=4,
                     qkv_bias=True,
                     drop_rate=0.,
                     attn_drop_rate=0.,
                     drop_path_rate=0.,
                     act_cfg=dict(type='GELU'),
                     norm_cfg=dict(type='LN', eps=1e-6),
                     pretrained=None,
                     init_cfg=None,
                     with_cp=False):
            super(MixVisionTransformer, self).__init__(init_cfg=init_cfg)
            if pretrained is not None:
                self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
            self.embed_dims = embed_dims
            self.num_stages = num_stages
            self.num_layers = num_layers
            self.num_heads = num_heads
            self.patch_sizes = patch_sizes
            self.strides = strides
            self.sr_ratios = sr_ratios
            self.with_cp = with_cp
            assert num_stages == len(num_layers) == len(num_heads) \
                   == len(patch_sizes) == len(strides) == len(sr_ratios)
            self.out_indices = out_indices
            assert max(out_indices) < self.num_stages
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))]
            cur = 0
            self.layers = ModuleList()
            current_in_channels = in_channels
            self.out_channels = []
            for i, num_layer in enumerate(num_layers):
                embed_dims_i = self.embed_dims * self.num_heads[i]
                self.out_channels.append(embed_dims_i)
                patch_embed = _PatchEmbed(
                    in_channels=current_in_channels,
                    embed_dims=embed_dims_i,
                    kernel_size=patch_sizes[i],
                    stride=strides[i],
                    padding=patch_sizes[i] // 2,
                    norm_cfg=norm_cfg)
                stage_layers = ModuleList([
                    TransformerEncoderLayer(
                        embed_dims=embed_dims_i,
                        num_heads=num_heads[i],
                        feedforward_channels=mlp_ratio * embed_dims_i,
                        drop_rate=drop_rate,
                        attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[cur + idx],
                        qkv_bias=qkv_bias,
                        act_cfg=act_cfg,
                        norm_cfg=norm_cfg,
                        with_cp=with_cp,
                        sr_ratio=sr_ratios[i]) for idx in range(num_layer)
                ])
                current_in_channels = embed_dims_i
                norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
                self.layers.append(ModuleList([patch_embed, stage_layers, norm]))
                cur += num_layer

        def init_weights(self):
            if self.init_cfg is None:
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        trunc_normal_init(m, std=.02, bias=0.)
                    elif isinstance(m, nn.LayerNorm):
                        constant_init(m, val=1.0, bias=0.)
                    elif isinstance(m, nn.Conv2d):
                        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        fan_out //= m.groups
                        normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
            else:
                super(MixVisionTransformer, self).init_weights()

        def forward(self, x):
            outs = []
            for i, layer_group in enumerate(self.layers):
                patch_embed, stage_blocks, norm_layer = layer_group
                x, hw_shape = patch_embed(x)
                for block in stage_blocks:
                    x = block(x, hw_shape)
                x = norm_layer(x)
                x = nlc_to_nchw(x, hw_shape)
                if i in self.out_indices:
                    outs.append(x)
            return outs

class PlaceholderMiT(nn.Module):
    def __init__(self, model_name_placeholder="mit_b0", pretrained_placeholder=False, in_chans_placeholder=1, **kwargs_ph):
        super().__init__()
        s1_out, s2_out, s3_out, s4_out = 32, 64, 160, 256
        self.feature_channels = [s1_out, s2_out, s3_out, s4_out]
        if 'mit_b0' in model_name_placeholder.lower():
            s1_out, s2_out, s3_out, s4_out = 32, 64, 160, 256
            self.feature_channels = [s1_out, s2_out, s3_out, s4_out]
        elif 'mit_b1' in model_name_placeholder.lower():
            s1_out, s2_out, s3_out, s4_out = 64, 128, 320, 512
            self.feature_channels = [s1_out, s2_out, s3_out, s4_out]
        elif 'mit_b2' in model_name_placeholder.lower():
            s1_out, s2_out, s3_out, s4_out = 64, 128, 320, 512
            self.feature_channels = [s1_out, s2_out, s3_out, s4_out]
        elif 'mit_b3' in model_name_placeholder.lower():
            s1_out, s2_out, s3_out, s4_out = 64, 128, 320, 512
            self.feature_channels = [s1_out, s2_out, s3_out, s4_out]
        elif 'mit_b4' in model_name_placeholder.lower():
            s1_out, s2_out, s3_out, s4_out = 64, 128, 320, 512
            self.feature_channels = [s1_out, s2_out, s3_out, s4_out]
        elif 'mit_b5' in model_name_placeholder.lower():
            s1_out, s2_out, s3_out, s4_out = 64, 128, 320, 512
            self.feature_channels = [s1_out, s2_out, s3_out, s4_out]
        else:
            s1_out, s2_out, s3_out, s4_out = 64, 128, 256, 512
            self.feature_channels = [s1_out, s2_out, s3_out, s4_out]
        self.conv_s1 = nn.Sequential(nn.Conv2d(in_chans_placeholder, s1_out, 7, 4, 3), nn.ReLU())
        self.conv_s2 = nn.Sequential(nn.Conv2d(s1_out, s2_out, 3, 2, 1), nn.ReLU())
        self.conv_s3 = nn.Sequential(nn.Conv2d(s2_out, s3_out, 3, 2, 1), nn.ReLU())
        self.conv_s4 = nn.Sequential(nn.Conv2d(s3_out, s4_out, 3, 2, 1), nn.ReLU())

    def forward(self, x):
        s1 = self.conv_s1(x)
        s2 = self.conv_s2(s1)
        s3 = self.conv_s3(s2)
        s4 = self.conv_s4(s3)
        return [s1, s2, s3, s4]

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("LayerNorm") != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.constant_(m.weight.data, 1.0)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)

class MiTEncoder(nn.Module):
    def __init__(self, model_name='mit_b0', pretrained_path=None, in_chans=1):
        super().__init__()
        self.model_name = model_name
        self.in_chans = in_chans
        if not MMSeg_AVAILABLE or MixVisionTransformer is None:
            self.encoder = PlaceholderMiT(model_name_placeholder=self.model_name,
                                          pretrained_placeholder=bool(pretrained_path),
                                          in_chans_placeholder=self.in_chans)
            self.feature_channels = self.encoder.feature_channels
            return
        config_common = dict(
            in_channels=in_chans,
            patch_sizes=[7, 3, 3, 3],
            strides=[4, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            act_cfg=dict(type='GELU'),
            norm_cfg=dict(type='LN', eps=1e-6),
            drop_rate=0.0,
            attn_drop_rate=0.0,
            with_cp=False
        )
        if model_name == 'mit_b0':
            base_embed_dims = 32
            config = dict(**config_common, embed_dims=base_embed_dims, num_stages=4, num_layers=[2, 2, 2, 2],
                          num_heads=[1, 2, 5, 8], drop_path_rate=0.1)
            self.feature_channels = [ch * base_embed_dims for ch in config['num_heads']]
        elif model_name == 'mit_b1':
            base_embed_dims = 64
            config = dict(**config_common, embed_dims=base_embed_dims, num_stages=4, num_layers=[2, 2, 2, 2],
                          num_heads=[1, 2, 5, 8], drop_path_rate=0.1)
            self.feature_channels = [ch * base_embed_dims for ch in config['num_heads']]
        elif model_name == 'mit_b2':
            base_embed_dims = 64
            config = dict(**config_common, embed_dims=base_embed_dims, num_stages=4, num_layers=[3, 4, 6, 3],
                          num_heads=[1, 2, 5, 8], drop_path_rate=0.1)
            self.feature_channels = [ch * base_embed_dims for ch in config['num_heads']]
        elif model_name == 'mit_b3':
            base_embed_dims = 64
            config = dict(**config_common, embed_dims=base_embed_dims, num_stages=4, num_layers=[3, 4, 18, 3],
                          num_heads=[1, 2, 5, 8], drop_path_rate=0.1)
            self.feature_channels = [ch * base_embed_dims for ch in config['num_heads']]
        elif model_name == 'mit_b4':
            base_embed_dims = 64
            config = dict(**config_common, embed_dims=base_embed_dims, num_stages=4, num_layers=[3, 8, 27, 3],
                          num_heads=[1, 2, 5, 8], drop_path_rate=0.1)
            self.feature_channels = [ch * base_embed_dims for ch in config['num_heads']]
        elif model_name == 'mit_b5':
            base_embed_dims = 64
            config = dict(**config_common, embed_dims=base_embed_dims, num_stages=4, num_layers=[3, 6, 40, 3],
                          num_heads=[1, 2, 5, 8], drop_path_rate=0.1)
            self.feature_channels = [ch * base_embed_dims for ch in config['num_heads']]
        else:
            raise ValueError(f"Unknown MiT model_name: {model_name}")
        try:
            init_cfg_dict = None
            if pretrained_path and os.path.exists(pretrained_path):
                init_cfg_dict = dict(type='Pretrained', checkpoint=pretrained_path)
            config['init_cfg'] = init_cfg_dict
            self.encoder = MixVisionTransformer(**config)
            if hasattr(self.encoder, 'out_channels') and self.encoder.out_channels != self.feature_channels:
                self.feature_channels = self.encoder.out_channels
        except Exception as e_instantiate:
            import traceback
            traceback.print_exc()
            self.encoder = PlaceholderMiT(model_name_placeholder=self.model_name,
                                          pretrained_placeholder=bool(pretrained_path),
                                          in_chans_placeholder=self.in_chans)
            self.feature_channels = self.encoder.feature_channels

    def forward(self, x):
        return self.encoder(x)

class SFFModule(nn.Module):
    def __init__(self, global_channels, local_channels, intermediate_channels=64, out_channels=None):
        super().__init__()
        if out_channels is None: out_channels = local_channels
        self.crg = nn.Conv2d(global_channels, intermediate_channels, 1,
                             bias=False) if global_channels != intermediate_channels else nn.Identity()
        self.crl = nn.Conv2d(local_channels, intermediate_channels, 1,
                             bias=False) if local_channels != intermediate_channels else nn.Identity()
        self.fconv = nn.Sequential(nn.Conv2d(intermediate_channels * 2, intermediate_channels, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(intermediate_channels), nn.ReLU(True),
                                   nn.Conv2d(intermediate_channels, intermediate_channels, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(intermediate_channels), nn.ReLU(True))
        self.aconv = nn.Conv2d(intermediate_channels, 2, 1, bias=True)
        self.finalc = nn.Conv2d(intermediate_channels, out_channels, 3, 1, 1,
                                bias=False) if intermediate_channels != out_channels else nn.Identity()

    def forward(self, global_feat, local_feat):
        if local_feat.shape[2:] != global_feat.shape[2:]:
            local_feat = F.interpolate(local_feat, size=global_feat.shape[2:], mode='bilinear', align_corners=False)
        gfr, lfr = self.crg(global_feat), self.crl(local_feat)
        fif = torch.cat((gfr, lfr), 1)
        pf = self.fconv(fif)
        am = torch.sigmoid(self.aconv(pf))
        ag, al = am[:, 0:1], am[:, 1:2]
        attg, attl = gfr * ag, lfr * al
        hf = attg + attl
        return self.finalc(hf)

class LightweightDecoder(nn.Module):
    def __init__(self, encoder_feature_channels, decoder_stage_channels, final_out_channels=1,
                 sff_intermediate_channels=64):
        super().__init__()
        self.ec = encoder_feature_channels
        self.dc = decoder_stage_channels
        if not self.ec or not isinstance(self.ec, list) or len(self.ec) != 4:
            raise ValueError(f"encoder_feature_channels must be a list of 4 elements. Got: {self.ec}")
        if not self.dc or not isinstance(self.dc, list) or len(self.dc) != 4:
            raise ValueError(f"decoder_stage_channels ({self.dc}) must have a length of 4.")
        self.bottleneck_conv = nn.Sequential(nn.Conv2d(self.ec[-1], self.dc[0], kernel_size=3, padding=1, bias=False),
                                             nn.BatchNorm2d(self.dc[0]), nn.ReLU(inplace=True))
        self.upsample_sff_stages = nn.ModuleList()
        current_decoder_input_ch = self.dc[0]
        for i in range(len(self.ec) - 1):
            encoder_skip_ch = self.ec[len(self.ec) - 2 - i]
            sff_output_ch = self.dc[i + 1]
            self.upsample_sff_stages.append(
                SFFModule(global_channels=encoder_skip_ch, local_channels=current_decoder_input_ch,
                          intermediate_channels=sff_intermediate_channels, out_channels=sff_output_ch))
            current_decoder_input_ch = sff_output_ch
        final_decoder_input_channels = self.dc[-1]
        self.final_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.final_convs = nn.Sequential(
            nn.Conv2d(final_decoder_input_channels, max(final_decoder_input_channels // 2, final_out_channels * 2, 16),
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(max(final_decoder_input_channels // 2, final_out_channels * 2, 16)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(final_decoder_input_channels // 2, final_out_channels * 2, 16), final_out_channels,
                      kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, encoder_features_list):
        if not isinstance(encoder_features_list, list) or len(encoder_features_list) != 4:
            raise ValueError(
                f"LightweightDecoder.forward: encoder_features_list must be a list of 4 elements, got: {type(encoder_features_list)}")
        s1_feats, s2_feats, s3_feats, s4_feats = encoder_features_list
        current_features = self.bottleneck_conv(s4_feats)
        current_features = self.upsample_sff_stages[0](s3_feats, current_features)
        current_features = self.upsample_sff_stages[1](s2_feats, current_features)
        current_features = self.upsample_sff_stages[2](s1_feats, current_features)
        output = self.final_upsample(current_features)
        output = self.final_convs(output)
        return output

class GeneratorMiTDecoder(nn.Module):
    def __init__(self, in_chans=1, out_chans=1, mit_model_name='mit_b0', mit_pretrained_path=None,
                 decoder_channels=None, sff_intermediate_channels=64):
        super().__init__()
        self.encoder = MiTEncoder(model_name=mit_model_name, pretrained_path=mit_pretrained_path, in_chans=in_chans)
        actual_encoder_channels = self.encoder.feature_channels
        if decoder_channels is None:
            dc_reversed = [c // 2 if c > 32 else c for c in reversed(actual_encoder_channels)]
            decoder_channels = [max(c, 16) for c in dc_reversed]
        self.decoder = LightweightDecoder(encoder_feature_channels=actual_encoder_channels,
                                          decoder_stage_channels=decoder_channels, final_out_channels=out_chans,
                                          sff_intermediate_channels=sff_intermediate_channels)

    def forward(self, x):
        encoder_features = self.encoder(x)
        return self.decoder(encoder_features)

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, use_cbam=False, use_refinement=False, num_refinement_blocks=3):
        super(GeneratorUNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Sigmoid(),
        )
        self.use_cbam = use_cbam
        if use_cbam: self.cbam_up6 = CBAM(128); self.cbam_final_input = CBAM(128)
        self.use_refinement = use_refinement
        if use_refinement: self.refinement_blocks = nn.Sequential(
            *[ResBlock(out_channels) for _ in range(num_refinement_blocks)])

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        if self.use_cbam and hasattr(self, 'cbam_up6'): u6 = self.cbam_up6(u6)
        u7 = self.up7(u6, d1)
        final_input_features = u7
        if self.use_cbam and hasattr(self, 'cbam_final_input'): final_input_features = self.cbam_final_input(
            final_input_features)
        out = self.final_conv(final_input_features)
        if self.use_refinement and hasattr(self, 'refinement_blocks'): out = self.refinement_blocks(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x_ca = self.ca(x) * x
        return self.sa(x_ca) * x_ca

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                        nn.InstanceNorm2d(channels), nn.ReLU(inplace=True),
                                        nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                        nn.InstanceNorm2d(channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x): return self.relu(x + self.conv_block(x))