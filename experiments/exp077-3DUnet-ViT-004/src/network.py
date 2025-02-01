import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CFG


# ==================================================
# 3D U-Net (ViTエンコーダを利用)
# ==================================================
class Unet3D(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        # 例: CFG.encoder_dim["vit_tiny_patch16_224"] = [192,192,192]
        encoder_dim = CFG.encoder_dim[CFG.model_name]  # 実際には3段 (Stage0,1,2)

        # デコーダ3段
        decoder_dim = [128, 64, 32]

        self.encoder = encoder
        self.decoder = Decoder3d(
            in_channel=encoder_dim[-1],               # 最深部 (192)
            skip_channel=encoder_dim[:-1][::-1] + [0],# 例: [192,192,0]
            out_channel=decoder_dim,                  # [128,64,32]
        )
        # 出力チャネル: (クラス数+1) => 6種 + 1 = 7
        self.cls = nn.Conv3d(
            decoder_dim[-1], 
            len(CFG.particles_name) + 1,
            kernel_size=1
        )

    def forward(self, x):
        """
        x.shape: (B, D=16, C=1, H=64, W=64)
        例: B=2, D=16, C=1, H=64, W=64
        """
        b, d, c, h, w = x.shape
        assert c == 1, "ViT へは3チャンネル入力が必要 → (C=1→3に拡張)"

        # (B*D,1,H,W) => (B*D,3,H,W)
        x = x.reshape(b*d, c, h, w).expand(-1, 3, -1, -1)

        # ========== 2D ViTエンコーダ ==========
        # feats=[stage0, stage1, stage2]
        encode = encode2d_vit(
            timm_encoder=self.encoder,
            input2d=x,
            batch_size=b,
            depth_scaler=[1, 2, 2],  # 3段に対応
        )

        # ========== 3D デコーダ (3段) ==========
        last, decode = self.decoder(
            feature=encode[-1],               # 最深部 stage2
            skip=encode[:-1][::-1] + [None],  # [stage1, stage0, None]
            depth_scaling=[1, 2, 2],          # 3段
            spatial_scaling=[2, 2, 2],        # シンプルに空間は2倍ずつ
        )

        # Conv3d (32 → 7)
        cls_ = self.cls(last)

        # ========== 最終サイズを (D=16,H=64,W=64) に固定 ==========
        cls_ = F.interpolate(
            cls_, 
            size=(16, 64, 64), 
            mode="nearest"
        )

        return cls_


# ==================================================
# 2D Encoder: ViT
# ==================================================
def encode2d_vit(timm_encoder, input2d, batch_size, depth_scaler=[1, 2, 2]):
    """
    feats => 3段想定: stage0, stage1, stage2
    """
    # print("input2d.shape =", input2d.shape)  # =>[32, 3, 64, 64] ok
    # 224×224に拡大
    input2d = F.interpolate(input2d, size=(224,224), mode='bilinear', align_corners=False)
    # print("interpolated input2d.shape =", input2d.shape)  # => [32, 3, 224, 224] ok
    feats = timm_encoder(input2d)  # 3段 => feats=[f0, f1, f2]
    print("feats.shape =", feats.shape)  # => [32, 197, 192]

    encoded = []
    for i, f in enumerate(feats):
        ds = depth_scaler[i] if i < len(depth_scaler) else 1
        l_pooled_f, _ = aggregate_depth(f, batch_size, ds)
        encoded.append(l_pooled_f)
    return encoded


# ==================================================
# Depth方向を集約 (avg_pool3d)
# ==================================================
def aggregate_depth(x, batch_size, depth_scaling):
    """
    x: (batch*depth, channel, H, W) # [197, 192]
    => (b,d,c,H,W) => 3Dpool => ...
    """
    print("x.shape =", x.shape)  # => (197, 192)
    bd, c, h, w = x.shape
    d = bd // batch_size
    b = batch_size

    x = x.reshape(b, d, c, h, w)   
    x = x.permute(0, 2, 1, 3, 4)

    pooled_x = F.avg_pool3d(
        x,
        kernel_size=(depth_scaling,1,1),
        stride=(depth_scaling,1,1),
        padding=0,
    )
    # => (b,c,d//ds,h,w)

    # 次ステージ入力
    encoder2d_input = pooled_x.permute(0,2,1,3,4).reshape(-1,c,h,w)
    return pooled_x, encoder2d_input


# ==================================================
# 3D Decoder
# ==================================================
class Block3d(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channel + skip_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None, depth_scaling=1, spatial_scaling=2):
        """
        x: (B,C,D,H,W)
        """
        x = F.interpolate(
            x,
            scale_factor=(depth_scaling, spatial_scaling, spatial_scaling),
            mode="nearest"
        )

        if skip is not None:
            skip = F.interpolate(skip, size=x.shape[2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Decoder3d(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel):
        super().__init__()
        # 最深部チャネル -> decoder_dim[0]
        self.channel_transform = nn.Conv3d(in_channel, out_channel[0], kernel_size=1)

        i_channel = [out_channel[0]] + out_channel[:-1]  # [128,128] for 3 items => [128,128,64]
        s_channel = skip_channel                          # 例: [192,192,0]
        o_channel = out_channel                           # [128,64,32]

        self.block = nn.ModuleList()
        for ic, sc, oc in zip(i_channel, s_channel, o_channel):
            self.block.append(Block3d(ic, sc, oc))

    def forward(self, feature, skip, depth_scaling=[1,2,2], spatial_scaling=[2,2,2]):
        d = self.channel_transform(feature)

        decode = []
        for i, block in enumerate(self.block):
            s = skip[i] if i < len(skip) else None
            ds = depth_scaling[i] if i < len(depth_scaling) else 1
            ss = spatial_scaling[i] if i < len(spatial_scaling) else 1
            d = block(d, skip=s, depth_scaling=ds, spatial_scaling=ss)
            decode.append(d)

        return d, decode


# テスト例
if __name__ == "__main__":
    x = torch.randn(2, 16, 1, 64, 64).cuda()

    import timm
    encoder = timm.create_model(
        model_name=CFG.model_name,
        pretrained=False,
        in_chans=3,
        num_classes=0,
        global_pool="",
        features_only=True,
    ).cuda()

    model = Unet3D(encoder).cuda()
    out = model(x)
    print("out.shape =", out.shape)  # => (2, 7, 16, 64, 64)
