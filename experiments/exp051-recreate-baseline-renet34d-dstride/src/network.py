import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CFG
from icecream import ic


# ------------------------------------------------------------
# 追加: クロスアテンション用のモジュール
# ------------------------------------------------------------
class CrossDomainAttention(nn.Module):
    """
    ドメイン埋め込み(クエリ)と特徴マップ(K, V)でクロスアテンションを行い、
    特徴マップをドメインに応じて変換するサンプル例。

    ※ MultiheadAttentionは2D(RNN)ベースの形状を想定しているため、
      3D特徴マップは (B, C, D, H, W) -> (B, N, C) にreshapeして扱います。
    """

    def __init__(self, embed_dim=128, num_heads=4, num_domains=5):
        super().__init__()
        # ドメインIDを埋め込む => 埋め込み次元は特徴チャネル数と同じにしておくとシンプル
        self.domain_embedding = nn.Embedding(num_domains, embed_dim)

        # MultiHeadAttentionを利用
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Query projection: ドメイン埋め込みを Q に変換 (省略可: そのままでも可)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        # 簡易的に使うためのLayerNorm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, domain_idx):
        """
        x: (B, C, D, H, W)  => B=バッチ数, C=チャネル, D=深さ, H/W=高さ/幅
        domain_idx: (B,)    => 各バッチが属するドメインID
        """
        B, C, D, H, W = x.shape
        # reshape => (B, N, C), N = D*H*W
        x_reshaped = x.view(B, C, -1).permute(0, 2, 1)  # => (B, N, C)

        # (B,) -> (B, embed_dim) にドメイン埋め込みを適用
        domain_embeds = self.domain_embedding(domain_idx)  # => (B, C)
        query = self.q_proj(domain_embeds).unsqueeze(1)  # => (B, 1, C)

        # MultiheadAttention(batch_first=True):
        #   query: (B,   tgt_len=1,   embed_dim=C)
        #   key  : (B,   src_len=N,   embed_dim=C)
        #   value: (B,   src_len=N,   embed_dim=C)
        out, _ = self.attn(query, x_reshaped, x_reshaped)
        # out => (B, 1, C)

        # outをブロードキャストして x_reshaped と組み合わせる (簡単な加算ゲート例)
        out_expanded = out.expand(-1, x_reshaped.size(1), -1)  # => (B, N, C)
        x_reshaped = x_reshaped + self.norm(out_expanded)  # 残差接続 + LayerNorm

        # 元の形状 (B, C, D, H, W) に戻す
        x_reshaped = x_reshaped.permute(0, 2, 1)  # => (B, C, N)
        x_out = x_reshaped.view(B, C, D, H, W)

        return x_out


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
            self.cross_attn = CrossDomainAttention(
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

    model = Unet3D(encoder, num_domains=4)

    output = model(x, domain_idx=domain_idx)
    print("output.shape => ", output.shape)
    # => (B, #class + 1, D, H, W)