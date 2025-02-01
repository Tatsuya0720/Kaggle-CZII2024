import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CFG
from icecream import ic


class Unet3D(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        encoder_dim = CFG.encoder_dim[CFG.model_name]
        decoder_dim = [256, 128, 64, 32, 16]

        self.encoder = encoder
        self.decoder = Decoder3d(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1] + [0],
            out_channel=decoder_dim,
        )

        self.cls = nn.Conv3d(
            decoder_dim[-1], len(CFG.particles_name) + 1, kernel_size=1
        )

    def forward(self, x):
        b, d, c, h, w = x.shape
        assert c == 1
        x = x.reshape(b * d, c, h, w)
        input2d = x.expand(-1, 3, -1, -1)  # (b*d, 3, h, w)
        # ic(input2d.shape)

        # encode = encode2d_timm(
        #     self.encoder,
        #     batch_size=b,
        #     input2d=input2d,
        #     depth_scaler=[2, 2, 2, 2, 1],
        # )
        # ViTを使う場合は encode2d_vit に切り替える
        # 既存のResNetを使うなら encode2d_timm を使う
        if "vit" in CFG.model_name:
            encode = encode2d_vit(
                self.encoder,
                batch_size=b,
                input2d=input2d,
                depth_scaler=[2, 2, 2, 2, 1],
            )
        else:
            encode = encode2d_timm(
                self.encoder,
                batch_size=b,
                input2d=input2d,
                depth_scaler=[2, 2, 2, 2, 1],
            )
        last, decode = self.decoder(
            feature=encode[-1],
            skip=encode[:-1][::-1] + [None],
            depth_scaling=[1, 2, 2, 2, 2],
        )

        # ic(last.shape)

        cls_ = self.cls(last)

        # ic(cls_.shape)

        return cls_

def aggregate_depth(x, batch_size, depth_scaling):
    """
    x: (batch*depth, channel, height, width)
    """
    bd, c, h, w = x.shape
    d = bd // batch_size
    b = batch_size

    x = x.reshape(b, d, c, h, w)  # (batch, depth, channel, height, width)
    x = x.permute(0, 2, 1, 3, 4)  # (batch, channel, depth, height, width)

    pooled_x = F.avg_pool3d(
        x,
        kernel_size=(depth_scaling, 1, 1),
        stride=(depth_scaling, 1, 1),
        padding=0,
    )  # (batch, channel, depth//depth_scaling, height, width)

    encoder2d_input_tensor = pooled_x.permute(
        0, 2, 1, 3, 4
    )  # (b, depth//depth_scaling, c, height, width)
    encoder2d_input_tensor = encoder2d_input_tensor.reshape(
        -1, c, h, w
    )  # (b*depth//depth_scaling, c, height, width)

    # pooled_x.shape => (b, c, depth//depth_scaling, height, width)
    # encoder2d_input_tensor.shape => (b*depth//depth_scaling, c, height, width)
    return pooled_x, encoder2d_input_tensor


def encode2d_timm(timm_encoder, input2d, batch_size, depth_scaler=[2, 2, 2, 2, 1]):
    """
    input2d: (batch*depth, channel, height, width)
    """

    encoded = []
    layer_00_input = timm_encoder.conv1(input2d)
    layer_00_input = timm_encoder.bn1(layer_00_input)
    layer_00_input = timm_encoder.act1(layer_00_input)
    l0_pooled_f, l0_pooled_encoder_input = aggregate_depth(
        layer_00_input, batch_size, depth_scaler[0]
    )
    # ic(l0_pooled_f.shape)
    encoded.append(l0_pooled_f)

    # ######## layer1 ########
    layer01_input = F.avg_pool2d(l0_pooled_encoder_input, kernel_size=2, stride=2)
    # layer01_input => (b*depth//depth_scaling, c, h//2, w//2)
    layer01_output = timm_encoder.layer1(layer01_input)
    # layer01_output => (b*depth//depth_scaling, c, h//2, w//2)
    l1_pooled_f, l1_pooled_encoder_input = aggregate_depth(
        layer01_output, batch_size, depth_scaler[1]
    )
    encoded.append(l1_pooled_f)
    # ic(l1_pooled_f.shape)

    # ######## layer2 ########
    layer02_output = timm_encoder.layer2(l1_pooled_encoder_input)
    # layer02_output => (b*depth//depth_scaling, c, h//4, w//4)
    l2_pooled_f, l2_pooled_encoder_input = aggregate_depth(
        layer02_output, batch_size, depth_scaler[2]
    )
    encoded.append(l2_pooled_f)
    # ic(l2_pooled_f.shape)

    # ######## layer3 ########
    layer03_output = timm_encoder.layer3(l2_pooled_encoder_input)
    # layer03_output => (b*depth//depth_scaling, c, h//8, w//8)
    l3_pooled_f, l3_pooled_encoder_input = aggregate_depth(
        layer03_output, batch_size, depth_scaler[3]
    )
    encoded.append(l3_pooled_f)
    # ic(l3_pooled_f.shape)

    # ######## layer4 ########
    layer04_output = timm_encoder.layer4(l3_pooled_encoder_input)
    # layer04_output => (b*depth//depth_scaling, c, h//16, w//16)
    l4_pooled_f, l4_pooled_encoder_input = aggregate_depth(
        layer04_output, batch_size, depth_scaler[4]
    )
    encoded.append(l4_pooled_f)
    # ic(l4_pooled_f.shape)

    return encoded

def encode2d_vit(timm_encoder, input2d, batch_size, depth_scaler=[2, 2, 2, 2, 1]):
    input2d = F.interpolate(input2d, size=(224, 224), mode="bilinear", align_corners=False)
    feats = timm_encoder(input2d)

    encoded = []
    for i, f in enumerate(feats):
        l_pooled_f, _ = aggregate_depth(f, batch_size, depth_scaler[i])
        
        # エンコーダの出力チャネル数を変換
        if l_pooled_f.shape[1] != 384:  # デコーダが期待するチャネル数
            l_pooled_f = F.conv3d(l_pooled_f, torch.randn(384, l_pooled_f.shape[1], 1, 1, 1).cuda(), bias=None)
        
        encoded.append(l_pooled_f)
    return encoded

# 3d decoder
class Block3d(nn.Module):
    def __init__(
        self,
        in_channel,
        skip_channel,
        out_channel,
    ):
        super().__init__()
        # print(in_channel , skip_channel, out_channel,)
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

    def forward(self, x, skip=None, depth_scaling=2):
        x = F.interpolate(x, scale_factor=(depth_scaling, 2, 2), mode="nearest")
        if skip is not None:
            # スキップ接続の解像度を調整 (デコーダの特徴マップと一致させる)
            skip = F.interpolate(skip, size=x.shape[2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class Decoder3d(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel):
        super().__init__()
        # エンコーダ出力チャネルに応じて変換層を動的に作成
        self.channel_transform = nn.Conv3d(in_channel, out_channel[0], kernel_size=1)
        self.center = nn.Identity()

        i_channel = [out_channel[0]] + out_channel[:-1]
        s_channel = skip_channel
        o_channel = out_channel
        block = [Block3d(i, s, o) for i, s, o in zip(i_channel, s_channel, o_channel)]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip, depth_scaling=[2, 2, 2, 2, 2, 1]):
        feature = self.channel_transform(feature)
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):
            s = skip[i]
            d = block(d, s, depth_scaling[i])
            decode.append(d)
        last = d
        return last, decode

    
# test-code
if __name__ == "__main__":
    # test-code
    b, c, d, h, w = CFG.batch_size, 1, 96, 320, 320
    # input2d = torch.randn(b * d, c, h, w)
    x = torch.randn(b, d, c, h, w)

    encoder = timm.create_model(
        model_name=CFG.model_name,
        pretrained=True,
        in_chans=3,
        num_classes=0,
        global_pool="",
        features_only=True,
    )

    model = Unet3D(encoder)

    output = model(x)