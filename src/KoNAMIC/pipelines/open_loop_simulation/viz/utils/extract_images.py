import torch
from torch import Tensor
from KoNAMIC.koopman.models import VisionValForwardOutputs

def _as_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def extract_open_loop_pred_gt_imgs_2d(
        output_k: VisionValForwardOutputs
) -> tuple[Tensor, Tensor]:

    pred = output_k.pred.y_left
    gt = output_k.g_t.y_left
    pred_imgs, gt_imgs = extract_open_loop_pred_gt_imgs(pred, gt)
    return pred_imgs, gt_imgs


def extract_open_loop_pred_gt_imgs_3d(
        output_k: VisionValForwardOutputs
) -> tuple[Tensor, Tensor, Tensor, Tensor]:

    pred_left = output_k.pred.y_left
    gt_left = output_k.g_t.y_left
    pred_right = output_k.pred.y_right
    gt_right = output_k.g_t.y_right

    pred_imgs_left, gt_imgs_left = extract_open_loop_pred_gt_imgs(pred_left, gt_left)
    pred_imgs_right, gt_imgs_right = extract_open_loop_pred_gt_imgs(pred_right, gt_right)
    return pred_imgs_left, gt_imgs_left, pred_imgs_right, gt_imgs_right


def extract_open_loop_pred_gt_imgs(pred: Tensor, gt: Tensor) -> tuple[Tensor, Tensor]:
    pred = _as_numpy(pred)
    pred_imgs_left = pred[:, 0]  # (T,H,W)
    gt = _as_numpy(gt)
    gt_imgs_left = gt[:, 0]  # (T,H,W)
    return pred_imgs_left, gt_imgs_left