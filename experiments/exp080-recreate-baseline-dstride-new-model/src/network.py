import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CFG
from icecream import ic


# ------------------------------------------------------------
# 追加: クロスアテンション用のモジュール
# ------------------------------------------------------------
class CrossDomainFeatureAdapter(nn.Module):
    """
    3D特徴マップ (B, C, D, H, W) と ドメインID (domain_idx) を入力し、
    ドメイン埋め込みをチャネル次元に結合して 2D Conv で変換する例。
    
    - domain_embedding: nn.Embedding(num_domains, embed_dim)
        => (B,) のドメインIDから (B, embed_dim) のベクトルを取得
    - それを (B, embed_dim, D, H, W) に拡張し、元の特徴マップと結合
    - depth方向 D をバッチに畳み込む (B*D, C+embed_dim, H, W) に reshape
    - Conv2d で処理後、(B, C_out, D, H, W) に戻す
    - Conv後に正規化・活性化を挟むのが一般的
    """

    def __init__(self, embed_dim=128, num_heads=4, num_domains=5):
        super().__init__()
        
        # 埋め込み層: domain_idx -> (embed_dim,)
        self.domain_embedding = nn.Embedding(num_domains, 32)
        
        # (C_in + embed_dim) -> C_out の 1x1 Conv (2D)
        self.conv = nn.Conv2d(embed_dim+32,embed_dim, 
                              kernel_size=5, stride=1, padding=2)
        
        # Conv2dの後にかける正規化と活性化 (2D向け)
        # ここは BatchNorm2d/InstanceNorm2d/GroupNormなど好みに応じて
        self.norm = nn.BatchNorm2d(embed_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, domain_idx):
        """
        x: (B, C, D, H, W)
        domain_idx: (B,)  => 各バッチのドメインID (0 ~ num_domains-1)
        """
        B, C, D, H, W = x.shape
        
        # (B,) -> (B, embed_dim)
        domain_embeds = self.domain_embedding(domain_idx)  # => (B, embed_dim)
        
        # ドメインembeddingを (B, embed_dim, D, H, W) へ拡張
        #   unsqueeze -> (B, embed_dim, 1, 1, 1)
        #   expand    -> (B, embed_dim, D, H, W)
        domain_embeds = domain_embeds.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        domain_embeds = domain_embeds.expand(B, domain_embeds.shape[1], D, H, W)  
        
        # x と domain_embeds をチャネル方向 (dim=1) で結合
        # => (B, C + embed_dim, D, H, W)
        out = torch.cat([x, domain_embeds], dim=1)
        
        # depth方向 D をバッチ次元に畳み込む
        # => (B*D, C + embed_dim, H, W)
        out = out.transpose(1, 2).reshape(B * D, C + domain_embeds.shape[1], H, W)
        
        # Conv2d -> Norm -> Activation
        out = self.conv(out)     # => (B*D, out_channels, H, W)
        out = self.norm(out)     # => (B*D, out_channels, H, W)
        out = self.act(out)      # => (B*D, out_channels, H, W)
        
        # 形を (B, out_channels, D, H, W) に戻す
        out = out.reshape(B, D, out.shape[1], H, W).transpose(1, 2)
        
        return out


# ------------------------------------------------------------
# U-Net本体の定義
# ------------------------------------------------------------
class Unet3D(nn.Module):
    def __init__(self, encoder, num_domains=5):
        super().__init__()
        encoder_dim = CFG.encoder_dim[CFG.model_name]
        decoder_dim = [256, 128, 64, 32, 16]

        self.encoder = encoder
        self.decoder = Decoder3d(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1] + [0],
            out_channel=decoder_dim,
            num_domains=num_domains,  # 追加
        )

        # 出力 (クラス数 + 1)
        self.cls = nn.Conv3d(
            decoder_dim[-1], len(CFG.particles_name) + 1, kernel_size=1
        )

    def forward(self, x, domain_idx):
        """
        x: (B, D, C, H, W)
        domain_idx: (B,)  # => 各サンプルに対してドメインIDを付与
        """
        b, d, c, h, w = x.shape
        assert c == 1, "入力チャネルは1であることを想定"
        x = x.reshape(b * d, c, h, w)

        # timmエンコーダは3ch入力を想定しているので、(1 -> 3) 拡張
        input2d = x.expand(-1, 3, -1, -1)  # (B*d, 3, H, W)

        # -----------------------------
        # Encoder
        # -----------------------------
        encode = encode2d_timm(
            self.encoder,
            batch_size=b,
            input2d=input2d,
            depth_scaler=[2, 2, 2, 2, 1],
        )
        # encode: List of 5 要素, 各要素 => (B, C, D', H', W')

        # -----------------------------
        # Decoder (+ domain_idx)
        # -----------------------------
        last, decode = self.decoder(
            feature=encode[-1],
            skip=encode[:-1][::-1] + [None],
            depth_scaling=[1, 2, 2, 2, 2],
            domain_idx=domain_idx,  # 追加
        )

        # 最終出力 (B, #class, D, H, W)
        cls_ = self.cls(last)
        return cls_


# 3D デコーダブロック
class Block3d(nn.Module):
    def __init__(
        self,
        in_channel,
        skip_channel,
        out_channel,
        num_domains=4,
        use_attention=True,
    ):
        super().__init__()
        # ----------------------------------------------------
        # クロスアテンション用: ドメインベクトルを特徴と融合
        # ----------------------------------------------------
        self.use_attention = use_attention
        if use_attention:
            self.cross_attn = CrossDomainFeatureAdapter(
                embed_dim=in_channel + skip_channel, num_domains=num_domains
            )

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channel + skip_channel,
                out_channel,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention1 = nn.Identity()

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None, depth_scaling=2, domain_idx=None):
        # -----------------------------
        # 1) アップサンプリング
        # -----------------------------
        x = F.interpolate(x, scale_factor=(depth_scaling, 2, 2), mode="nearest")

        # -----------------------------
        # 2) skip と結合
        # -----------------------------
        if skip is not None:
            x = torch.cat([x, skip], dim=1)  # (B, in_channel + skip_channel, D, H, W)

        # -----------------------------
        # 3) ドメインクロスアテンション（任意）
        # -----------------------------
        if self.use_attention and (domain_idx is not None):
           x = self.cross_attn(x, domain_idx)  # (B, in_channel+skip_channel, D, H, W)

        x = self.attention1(x)  # (Identity のままなら処理なし)
        x = self.conv1(x)

        x = self.conv2(x)
        x = self.attention2(x)  # (Identity のままなら処理なし)
        return x


class Decoder3d(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel, num_domains=4):
        super().__init__()
        self.center = nn.Identity()

        i_channel = [in_channel] + out_channel[:-1]
        s_channel = skip_channel
        o_channel = out_channel

        # 各ブロックに cross-attn を仕込む
        block = [
            Block3d(
                in_c,
                s_c if s_c else 0,
                out_c,
                num_domains=num_domains,
                use_attention=True,
            )
            for in_c, s_c, out_c in zip(i_channel, s_channel, o_channel)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip, depth_scaling, domain_idx):
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):
            s = skip[i]
            d = block(d, skip=s, depth_scaling=depth_scaling[i], domain_idx=domain_idx)
            decode.append(d)
        last = d
        return last, decode


# ------------------------------------------------------------
# timmエンコーダ + 3Dプーリング
# ------------------------------------------------------------
def aggregate_depth(x, batch_size, depth_scaling):
    """
    x: (batch*depth, channel, height, width)
    => (b, c, d, h, w) に再構成し、depth方向に avg_pool3d
    """
    bd, c, h, w = x.shape
    d = bd // batch_size
    b = batch_size

    x = x.reshape(b, d, c, h, w)  # (B, D, C, H, W)
    x = x.permute(0, 2, 1, 3, 4)  # (B, C, D, H, W)

    pooled_x = F.avg_pool3d(
        x,
        kernel_size=(depth_scaling, 1, 1),
        stride=(depth_scaling, 1, 1),
        padding=0,
    )  # => (B, C, D//depth_scaling, H, W)

    encoder2d_input_tensor = pooled_x.permute(
        0, 2, 1, 3, 4
    )  # => (B, D//depth_scaling, C, H, W)
    encoder2d_input_tensor = encoder2d_input_tensor.reshape(
        -1, c, h, w
    )  # => (B*D//depth_scaling, C, H, W)

    return pooled_x, encoder2d_input_tensor


def encode2d_timm(timm_encoder, input2d, batch_size, depth_scaler=[2, 2, 2, 2, 1]):
    """
    input2d: (B*D, channel=3, H, W)
    """
    encoded = []

    # conv1 -> bn1 -> act1
    layer_00_input = timm_encoder.conv1(input2d)
    layer_00_input = timm_encoder.bn1(layer_00_input)
    layer_00_input = timm_encoder.act1(layer_00_input)
    l0_pooled_f, l0_pooled_encoder_input = aggregate_depth(
        layer_00_input, batch_size, depth_scaler[0]
    )
    encoded.append(l0_pooled_f)

    # layer1
    layer01_input = F.avg_pool2d(l0_pooled_encoder_input, kernel_size=2, stride=2)
    layer01_output = timm_encoder.layer1(layer01_input)
    l1_pooled_f, l1_pooled_encoder_input = aggregate_depth(
        layer01_output, batch_size, depth_scaler[1]
    )
    encoded.append(l1_pooled_f)

    # layer2
    layer02_output = timm_encoder.layer2(l1_pooled_encoder_input)
    l2_pooled_f, l2_pooled_encoder_input = aggregate_depth(
        layer02_output, batch_size, depth_scaler[2]
    )
    encoded.append(l2_pooled_f)

    # layer3
    layer03_output = timm_encoder.layer3(l2_pooled_encoder_input)
    l3_pooled_f, l3_pooled_encoder_input = aggregate_depth(
        layer03_output, batch_size, depth_scaler[3]
    )
    encoded.append(l3_pooled_f)

    # layer4
    layer04_output = timm_encoder.layer4(l3_pooled_encoder_input)
    l4_pooled_f, l4_pooled_encoder_input = aggregate_depth(
        layer04_output, batch_size, depth_scaler[4]
    )
    encoded.append(l4_pooled_f)

    return encoded


# ------------------------------------------------------------
# テストコード
# ------------------------------------------------------------
if __name__ == "__main__":
    b, c, d, h, w = CFG.batch_size, 1, 32, 128, 128  # 例
    x = torch.randn(b, d, c, h, w)

    # ドメインID (B,) 例: [0,0,1,1,2,3...]
    domain_idx = torch.randint(0, 4, size=(b,))

    encoder = timm.create_model(
        model_name=CFG.model_name,
        pretrained=True,
        in_chans=3,
        num_classes=0,
        global_pool="",
        features_only=True,
    )

    model = Unet3D(encoder, num_domains=5)

    output = model(x, domain_idx=domain_idx)
    print("output.shape => ", output.shape)
    # => (B, #class + 1, D, H, W)