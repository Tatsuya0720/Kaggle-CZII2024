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
        encoder_dim = CFG.encoder_dim[CFG.model_name]  # 例: [192,192,192]
        
        # デコーダ段数を増やす (例: 5段)
        # 段が増えるほど空間解像度を細かく取り戻しやすくなり、表現力向上が期待できる
        decoder_dim = [256, 128, 64, 32, 16]

        self.encoder = encoder
        self.decoder = Decoder3d(
            in_channel=encoder_dim[-1],        # 例: 192
            skip_channel=encoder_dim[:-1][::-1] + [0],  
            out_channel=decoder_dim,           # [256,128,64,32,16]
        )
        # クラス数+1 の出力チャネル
        self.cls = nn.Conv3d(
            decoder_dim[-1], 
            len(CFG.particles_name) + 1, 
            kernel_size=1
        )

    def forward(self, x):
        """
        x.shape: (B, D, C=1, H, W)
        例: B=2, D=16, C=1, H=64, W=64
        """
        b, d, c, h, w = x.shape
        assert c == 1, "ViTエンコーダには 3ch 入力が必要 => (C=1→3に展開)"

        # (B*D, 1, H, W) -> (B*D, 3, H, W)
        x = x.reshape(b*d, c, h, w).expand(-1, 3, -1, -1)

        # ========== 2D ViTエンコーダ ==========
        encode = encode2d_vit(
            timm_encoder=self.encoder,
            input2d=x,
            batch_size=b,
            # 例えば: feats=[stage0, stage1, stage2], なら [1,2,2]
            depth_scaler=[1, 2, 2],  
        )

        # ========== 3Dデコーダ (5段) ==========
        # 例えば: depth_scaling=[1,2,2,2,2], spatial_scaling=[2,2,2,2,2] などで段数を合わせる
        last, decode = self.decoder(
            feature=encode[-1],
            skip=encode[:-1][::-1] + [None],  # スキップ接続
            depth_scaling=[1, 2, 2, 2, 2],    # 深さ方向を順にアップ
            spatial_scaling=[2, 2, 2, 2, 2],  # 空間方向も各段階で2倍アップ
        )

        # 1×1 Conv3dでクラス数に変換
        cls_ = self.cls(last)

        # ========== ここで最終的にサイズ固定 (D=32, H=64, W=64) ==========
        cls_ = F.interpolate(
            cls_, 
            size=(32, 64, 64), 
            mode="nearest"
        )

        return cls_


# ==================================================
# 2D Encoder: ViT
# ==================================================
def encode2d_vit(timm_encoder, input2d, batch_size, depth_scaler=[1,2,2]):
    """
    - input2d: (B*D,3,H,W)
    - (224,224)に拡大しViTに通す -> feats=[feat0, feat1, feat2,...]
    - depth_scalerに従って深さ方向を段階的にavg_pool3dし、encoded[]に格納
    """
    # ViTに合わせて224×224へ
    input2d = F.interpolate(input2d, size=(224,224), mode='bilinear', align_corners=False)
    feats = timm_encoder(input2d)  # 例: stage数=3なら [feat0, feat1, feat2]

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
    x: (batch*depth, channel, H, W)
    => (b, d, c, H, W) => 3D pooling => (b, c, d//ds, H, W) => ...
    """
    bd, c, h, w = x.shape
    d = bd // batch_size
    b = batch_size

    x = x.reshape(b, d, c, h, w)   # => (b,d,c,H,W)
    x = x.permute(0, 2, 1, 3, 4)   # => (b,c,d,H,W)

    pooled_x = F.avg_pool3d(
        x,
        kernel_size=(depth_scaling,1,1),
        stride=(depth_scaling,1,1),
        padding=0,
    )
    # => (b, c, d//ds, H, W)

    # 次ステージ入力用に (b*d//ds, c, H, W) に戻す
    encoder2d_input = pooled_x.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
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
        self.attention1 = nn.Identity()

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None, depth_scaling=1, spatial_scaling=2):
        """
        x: (B,C,D,H,W)
        skip: 同上 (B,C_skip,D_skip,H_skip,W_skip)
        depth_scaling, spatial_scaling: 各段のアップサンプル倍率
        """
        x = F.interpolate(
            x, 
            scale_factor=(depth_scaling, spatial_scaling, spatial_scaling), 
            mode='nearest'
        )

        if skip is not None:
            skip = F.interpolate(skip, size=x.shape[2:], mode='nearest')
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class Decoder3d(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel):
        """
        ここでは "out_channel=[256,128,64,32,16]" => 5段
        skip_channel=[192,192,192, ...?, 0] => 長さ5にしておく
        """
        super().__init__()
        self.channel_transform = nn.Conv3d(in_channel, out_channel[0], kernel_size=1)
        self.center = nn.Identity()

        # 5段分ブロックを作る
        i_channel = [out_channel[0]] + out_channel[:-1]  # ex: [256,256,128,64,32]
        s_channel = skip_channel                          # ex: [192,192,192,...,0]
        o_channel = out_channel                           # [256,128,64,32,16]

        self.block = nn.ModuleList()
        for ic, sc, oc in zip(i_channel, s_channel, o_channel):
            self.block.append(Block3d(ic, sc, oc))

    def forward(
        self, 
        feature, 
        skip, 
        depth_scaling=[1,2,2,2,2], 
        spatial_scaling=[2,2,2,2,2]
    ):
        """
        depth_scaling, spatial_scaling の長さ=5 => 5段
        """
        d = self.channel_transform(feature)
        d = self.center(d)

        decode = []
        for i, block in enumerate(self.block):
            # skip[i]が足りなければNone
            s = skip[i] if i < len(skip) else None
            ds = depth_scaling[i] if i < len(depth_scaling) else 1
            ss = spatial_scaling[i] if i < len(spatial_scaling) else 1
            d = block(d, skip=s, depth_scaling=ds, spatial_scaling=ss)
            decode.append(d)

        return d, decode


# --------------------------------------------------
# テスト用コード
# --------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    # ダミー入力
    x = torch.randn(2, 16, 1, 64, 64).cuda()

    # ViT Encoder
    encoder = timm.create_model(
        model_name=CFG.model_name,  # 例: 'vit_tiny_patch16_224'
        pretrained=False,
        in_chans=3,
        num_classes=0,
        global_pool="",
        features_only=True,
    ).cuda()

    model = Unet3D(encoder).cuda()
    out = model(x)
    print("out.shape =", out.shape)
    # => たとえば (2,7,32,64,64)
