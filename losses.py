import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
import lpips
from piqa import ssim as piqa_ssim


class GradientLoss(nn.Module):
    def __init__(self, channels=1, loss_func_type='l1', device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = nn.L1Loss().to(self.device) if loss_func_type == 'l1' else nn.MSELoss().to(self.device)
        self.channels = channels

        kernel_x = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        kernel_y = np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])

        self.conv_x = nn.Conv2d(channels, channels, 3, 1, 1, bias=False, groups=channels)
        self.conv_y = nn.Conv2d(channels, channels, 3, 1, 1, bias=False, groups=channels)

        with torch.no_grad():
            kx = torch.tensor(kernel_x, dtype=torch.float32)
            ky = torch.tensor(kernel_y, dtype=torch.float32)
            self.conv_x.weight.data.copy_(kx.view(1, 1, 3, 3).repeat(self.channels, 1, 1, 1))
            self.conv_y.weight.data.copy_(ky.view(1, 1, 3, 3).repeat(self.channels, 1, 1, 1))

        self.conv_x.weight.requires_grad = False
        self.conv_y.weight.requires_grad = False
        self.conv_x = self.conv_x.to(self.device)
        self.conv_y = self.conv_y.to(self.device)

    def forward(self, pred, target):
        pred, target = pred.to(self.device), target.to(self.device)
        dx_pred = self.conv_x(pred)
        dy_pred = self.conv_y(pred)
        dx_gt = self.conv_x(target)
        dy_gt = self.conv_y(target)
        return self.loss(dx_pred, dx_gt) + self.loss(dy_pred, dy_gt)


class WeightedL1Loss(nn.Module):
    def __init__(self, channels=1, min_weight=1.0, max_weight=5.0, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_weight = min_weight
        self.max_weight = max_weight
        grad = GradientLoss(channels=channels, device=self.device)
        self.conv_x = grad.conv_x.to(self.device)
        self.conv_y = grad.conv_y.to(self.device)

    def forward(self, pred, target):
        pred, target = pred.to(self.device), target.to(self.device)
        dx = self.conv_x(target)
        dy = self.conv_y(target)
        grad_mag = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        bmin = grad_mag.view(grad_mag.size(0), -1).amin(1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        bmax = grad_mag.view(grad_mag.size(0), -1).amax(1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        norm = (grad_mag - bmin) / (bmax - bmin + 1e-8)
        weight = torch.clamp(1.0 + norm, self.min_weight, self.max_weight)
        return torch.mean(weight * torch.abs(pred - target))


class SurfaceNormalLoss(nn.Module):
    def __init__(self, loss_type='cosine', pixel_size_x=1.0, pixel_size_y=1.0, device=None, epsilon=1e-8):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_type = loss_type
        self.pixel_size_x = pixel_size_x
        self.pixel_size_y = pixel_size_y
        self.epsilon = epsilon

        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        self.conv_x = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        self.conv_y = nn.Conv2d(1, 1, 3, 1, 1, bias=False)

        with torch.no_grad():
            kx = torch.tensor(kernel_x, dtype=torch.float32)
            ky = torch.tensor(kernel_y, dtype=torch.float32)
            self.conv_x.weight.data.copy_(kx.view(1, 1, 3, 3))
            self.conv_y.weight.data.copy_(ky.view(1, 1, 3, 3))

        self.conv_x = self.conv_x.to(self.device)
        self.conv_y = self.conv_y.to(self.device)
        self.conv_x.weight.requires_grad = False
        self.conv_y.weight.requires_grad = False

    def forward(self, pred, target):
        pred, target = pred.to(self.device), target.to(self.device)
        dx_pred, dy_pred = self.conv_x(pred), self.conv_y(pred)
        dx_gt, dy_gt = self.conv_x(target), self.conv_y(target)

        normal_pred = torch.cat([-dx_pred * self.pixel_size_x, -dy_pred * self.pixel_size_y, torch.ones_like(dx_pred)], dim=1)
        normal_gt = torch.cat([-dx_gt * self.pixel_size_x, -dy_gt * self.pixel_size_y, torch.ones_like(dx_gt)], dim=1)

        normal_pred = F.normalize(normal_pred, dim=1, eps=self.epsilon)
        normal_gt = F.normalize(normal_gt, dim=1, eps=self.epsilon)

        if self.loss_type == 'cosine':
            return torch.mean(1.0 - torch.sum(normal_pred * normal_gt, dim=1, keepdim=True))
        elif self.loss_type == 'l1':
            return F.l1_loss(normal_pred, normal_gt)
        elif self.loss_type == 'l2':
            return F.mse_loss(normal_pred, normal_gt)
        else:
            raise ValueError("Unknown surface normal loss type")


class PerceptualLoss(nn.Module):
    def __init__(self, net='alex', device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net=net).eval().to(self.device)

    def forward(self, pred, target):
        pred = pred * 2 - 1 if pred.min() >= 0 else pred
        target = target * 2 - 1 if target.min() >= 0 else target
        return self.model(pred.to(self.device), target.to(self.device)).mean()


class SSIMLoss(nn.Module):
    def __init__(self, data_range=1.0, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ssim = piqa_ssim.SSIM(data_range=data_range).eval().to(self.device)

    def forward(self, pred, target):
        return 1.0 - self.ssim(pred.to(self.device), target.to(self.device))


class BerHuLoss(nn.Module):
    def __init__(self, delta_ratio=0.2, device=None):
        super().__init__()
        self.delta_ratio = delta_ratio
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, pred, target):
        pred, target = pred.to(self.device), target.to(self.device)
        abs_err = torch.abs(pred - target)
        delta = self.delta_ratio * torch.max(abs_err).clamp(min=1e-8)
        loss = torch.where(abs_err <= delta, abs_err, (abs_err ** 2 + delta ** 2) / (2 * delta))
        return loss.mean()


class TVLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.tv_loss_internal_factor = weight

    def forward(self, x):
        loss = (
            torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) +
            torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        )
        return self.tv_loss_internal_factor * loss


class SecondOrderGradientLoss(nn.Module):
    def __init__(self, channels=1, loss_func_type='l1', device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = nn.L1Loss().to(self.device) if loss_func_type == 'l1' else nn.MSELoss().to(self.device)
        self.channels = channels

        kernel_xx = np.array([[1, -2, 1],
                              [2, -4, 2],
                              [1, -2, 1]])

        kernel_yy = np.array([[1, 2, 1],
                              [-2, -4, -2],
                              [1, 2, 1]])

        self.conv_xx = nn.Conv2d(channels, channels, 3, 1, 1, bias=False, groups=channels)
        self.conv_yy = nn.Conv2d(channels, channels, 3, 1, 1, bias=False, groups=channels)

        with torch.no_grad():
            kxx = torch.tensor(kernel_xx, dtype=torch.float32)
            kyy = torch.tensor(kernel_yy, dtype=torch.float32)
            self.conv_xx.weight.data.copy_(kxx.view(1, 1, 3, 3).repeat(self.channels, 1, 1, 1))
            self.conv_yy.weight.data.copy_(kyy.view(1, 1, 3, 3).repeat(self.channels, 1, 1, 1))

        self.conv_xx.weight.requires_grad = False
        self.conv_yy.weight.requires_grad = False
        self.conv_xx = self.conv_xx.to(self.device)
        self.conv_yy = self.conv_yy.to(self.device)

    def forward(self, pred, target):
        pred, target = pred.to(self.device), target.to(self.device)
        dxx_pred = self.conv_xx(pred)
        dyy_pred = self.conv_yy(pred)
        dxx_gt = self.conv_xx(target)
        dyy_gt = self.conv_yy(target)
        return self.loss(dxx_pred, dxx_gt) + self.loss(dyy_pred, dyy_gt)


class RangeLoss(nn.Module):
    def __init__(self, reduction='mean', device=None):
        super().__init__()
        self.reduction = reduction
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, pred, target):
        pred = pred.to(self.device)
        target = target.to(self.device)

        dims = tuple(range(2, pred.dim()))
        p_min = torch.amin(pred, dim=dims)
        p_max = torch.amax(pred, dim=dims)
        t_min = torch.amin(target, dim=dims)
        t_max = torch.amax(target, dim=dims)

        loss = torch.abs(p_min - t_min) + torch.abs(p_max - t_max)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class L1InMetersFromCSV(nn.Module):
    def __init__(self, csv_path, eps=1e-6, device=None, strict=True):
        super().__init__()
        self.csv_path = str(csv_path)
        self.eps = eps
        self.strict = strict
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.id2range = self._load_csv(self.csv_path)
        self._batch_names = None

    def _load_csv(self, csv_path):
        id2r = {}
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                _id = str(row["id"])
                zmin = float(row["z_min"]); zmax = float(row["z_max"])
                id2r[_id] = (zmin, zmax)
        if not id2r:
            raise RuntimeError(f"CSV is empty or failed to read: {csv_path}")
        return id2r

    def set_batch_names(self, names):
        out = []
        for n in names:
            if isinstance(n, (bytes, bytearray)):
                out.append(n.decode('utf-8', errors='ignore'))
            else:
                out.append(str(n))
        self._batch_names = out

    def forward(self, pred01, target01):
        if self._batch_names is None:
            raise AssertionError("Please call set_batch_names(h5_names) before this step.")

        B = pred01.size(0)
        if len(self._batch_names) != B:
            raise ValueError(f"Number of batch names ({len(self._batch_names)}) does not match batch size ({B}).")

        zmins, zmaxs, missing = [], [], []
        for i, nm in enumerate(self._batch_names):
            pair = self.id2range.get(nm)
            if pair is None:
                missing.append((i, nm))
                if not self.strict:
                    zmins.append(0.0); zmaxs.append(1.0)
                    continue
                zmins.append(0.0); zmaxs.append(1.0)
            else:
                zmin, zmax = pair
                zmins.append(zmin); zmaxs.append(zmax)

        if missing and self.strict:
            head = "\n".join([f"  idx={i}, id='{nm}'" for i, nm in missing[:8]])
            raise KeyError(f"The following samples were not found in the CSV for z_min/z_max (first 8):\n{head}\n"
                           f"Total missing: {len(missing)}. Set strict=False to ignore.")

        z_min = torch.tensor(zmins, dtype=torch.float32, device=pred01.device).view(B,1,1,1)
        z_max = torch.tensor(zmaxs, dtype=torch.float32, device=pred01.device).view(B,1,1,1)
        z_rng = (z_max - z_min).clamp_min(self.eps)

        err_m = torch.abs(pred01 - target01) * z_rng
        return err_m.mean()


def get_loss_functions(csv_path_for_l1_meters, device=None, channels=1, berhu_delta_factor=0.2, tv_weight_factor=1.0):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        'GAN': nn.BCEWithLogitsLoss().to(device),
        'Pixelwise': nn.L1Loss().to(device),
        'Grad1': GradientLoss(channels=channels, device=device),
        'Grad2': SecondOrderGradientLoss(channels=channels, device=device),
        'Perceptual': PerceptualLoss(device=device),
        'WeightedL1': WeightedL1Loss(channels=channels, device=device),
        'SSIM': SSIMLoss(device=device),
        'BerHu': BerHuLoss(delta_ratio=berhu_delta_factor, device=device),
        'SurfaceNormal': SurfaceNormalLoss(device=device),
        'TV': TVLoss(weight=tv_weight_factor).to(device),
        'Range': RangeLoss(device=device),
        'L1': L1InMetersFromCSV(
            csv_path=csv_path_for_l1_meters,
            device=device,
            strict=True
        )
    }