"""
PointPillars encoder (pillar feature net + scatter to BEV pseudo-image) +
a small conv backbone + a CenterPoint-style anchor-free detection head.

Why anchor-free: real PointPillars uses 2 fixed-rotation anchors/class and
matches them to GT via rotated-BEV IoU — correct but a lot of finicky code.
A center-heatmap head (CenterNet/CenterPoint-style) needs only "which grid
cell is a box center" to build training targets, which is far simpler to
get right, and is what modern production 3D detectors actually use.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import pp_config as cfg


class PillarFeatureNet(nn.Module):
    def __init__(self, in_channels=cfg.PILLAR_FEAT_DIM, out_channels=cfg.PILLAR_OUT_CHANNELS):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, pillars, num_points):
        # pillars: (B, P, N, C_in), num_points: (B, P)
        B, P, N, C = pillars.shape
        x = self.linear(pillars.view(-1, C))
        x = self.bn(x)
        x = F.relu(x)
        x = x.view(B, P, N, -1)

        idx = torch.arange(N, device=pillars.device).view(1, 1, N)
        valid = (idx < num_points.unsqueeze(-1)).unsqueeze(-1).float()  # (B,P,N,1)
        x = x * valid
        x = x.max(dim=2)[0]  # (B, P, out_channels)
        return x


class PointPillarsScatter(nn.Module):
    def __init__(self, grid_h=cfg.GRID_H, grid_w=cfg.GRID_W, channels=cfg.PILLAR_OUT_CHANNELS):
        super().__init__()
        self.grid_h, self.grid_w, self.channels = grid_h, grid_w, channels

    def forward(self, pillar_features, coords, n_pillars):
        B = pillar_features.shape[0]
        canvas = torch.zeros(
            B, self.channels, self.grid_h, self.grid_w,
            device=pillar_features.device, dtype=pillar_features.dtype,
        )
        for b in range(B):
            n = int(n_pillars[b])
            if n == 0:
                continue
            rows = coords[b, :n, 0]
            cols = coords[b, :n, 1]
            canvas[b, :, rows, cols] = pillar_features[b, :n].t()
        return canvas


def conv_bn_relu(c_in, c_out, stride=1):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class Backbone(nn.Module):
    """Full-res block -> /2 block -> /4 block -> upsample /4 back to /2 and concat."""

    def __init__(self, c_in=cfg.PILLAR_OUT_CHANNELS):
        super().__init__()
        self.block1 = nn.Sequential(*[conv_bn_relu(c_in, c_in) for _ in range(3)])  # full res

        self.block2_down = conv_bn_relu(c_in, 128, stride=2)  # /2
        self.block2 = nn.Sequential(*[conv_bn_relu(128, 128) for _ in range(4)])

        self.block3_down = conv_bn_relu(128, 256, stride=2)  # /4
        self.block3 = nn.Sequential(*[conv_bn_relu(256, 256) for _ in range(4)])

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        c1 = self.block1(x)
        c2 = self.block2(self.block2_down(c1))       # /2, 128ch — matches FEAT_H/FEAT_W
        c3 = self.block3(self.block3_down(c2))        # /4, 256ch
        c3_up = self.up3(c3)                           # /2, 128ch (may be off by a pixel from rounding)
        if c3_up.shape[-2:] != c2.shape[-2:]:
            c3_up = F.interpolate(c3_up, size=c2.shape[-2:], mode="nearest")
        return torch.cat([c2, c3_up], dim=1)            # /2, 256ch


class CenterHead(nn.Module):
    def __init__(self, c_in=256, c_mid=64):
        super().__init__()
        self.shared = conv_bn_relu(c_in, c_mid)
        self.heatmap = nn.Conv2d(c_mid, 1, 1)
        self.offset = nn.Conv2d(c_mid, 2, 1)
        self.z = nn.Conv2d(c_mid, 1, 1)
        self.dim = nn.Conv2d(c_mid, 3, 1)
        self.rot = nn.Conv2d(c_mid, 2, 1)
        # Bias the heatmap head negative at init so training starts from
        # "predict background everywhere", standard CenterNet trick.
        nn.init.constant_(self.heatmap.bias, -2.19)

    def forward(self, x):
        f = self.shared(x)
        return {
            "heatmap": self.heatmap(f),  # logits
            "offset": self.offset(f),
            "z": self.z(f),
            "dim": self.dim(f),
            "rot": self.rot(f),
        }


class PointPillarsCenterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pfn = PillarFeatureNet()
        self.scatter = PointPillarsScatter()
        self.backbone = Backbone()
        self.head = CenterHead()

    def forward(self, pillars, coords, num_points, n_pillars):
        feats = self.pfn(pillars, num_points)
        canvas = self.scatter(feats, coords, n_pillars)
        backbone_feat = self.backbone(canvas)
        return self.head(backbone_feat)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def focal_loss(pred_logits, gt_heatmap):
    """Penalty-reduced pixelwise focal loss (CenterNet)."""
    pred = torch.sigmoid(pred_logits).clamp(1e-4, 1 - 1e-4)
    pos_mask = (gt_heatmap == 1).float()
    neg_mask = (gt_heatmap < 1).float()
    neg_weights = torch.pow(1 - gt_heatmap, 4)

    pos_loss = -torch.log(pred) * torch.pow(1 - pred, 2) * pos_mask
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_mask

    num_pos = pos_mask.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def masked_l1_loss(pred, target, mask):
    mask = mask.expand_as(pred)
    denom = mask.sum().clamp(min=1)
    return (torch.abs(pred - target) * mask).sum() / denom


def compute_loss(pred, batch):
    l_hm = focal_loss(pred["heatmap"], batch["heatmap"])
    l_off = masked_l1_loss(pred["offset"], batch["offset"], batch["mask"])
    l_z = masked_l1_loss(pred["z"], batch["z"], batch["mask"])
    l_dim = masked_l1_loss(pred["dim"], batch["dim"], batch["mask"])
    l_rot = masked_l1_loss(pred["rot"], batch["rot"], batch["mask"])

    total = l_hm + l_off + l_z + l_dim + l_rot
    return total, {
        "heatmap": l_hm.item(), "offset": l_off.item(), "z": l_z.item(),
        "dim": l_dim.item(), "rot": l_rot.item(), "total": total.item(),
    }
