# metrics.py

"""
Custom metrics for 3D scene understanding and motion planning, updated for modern
torchmetrics and PyTorch Lightning APIs.

This module includes:
- IntersectionOverUnion: A standard semantic segmentation metric.
- PanopticMetric: A complex metric for evaluating panoptic segmentation quality with
  temporal consistency checks.
- PlanningMetric: A metric for evaluating motion planning trajectories against ground
  truth and semantic maps for collisions and L2 distance.
"""

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import numpy as np
from torchmetrics import Metric
from torchmetrics.functional.classification import multiclass_stat_scores
from skimage.draw import polygon

from stp3.utils.tools import gen_dx_bx
from stp3.utils.geometry import calculate_birds_eye_view_parameters


class IntersectionOverUnion(Metric):
    """
    Computes Intersection-over-Union (IoU) for semantic segmentation.

    This metric is a wrapper around torchmetrics' functional `multiclass_stat_scores`
    and calculates the final IoU score.
    """
    # By default, all metrics have full_state_update=False. This is usually the desired
    # behavior for performance reasons.
    full_state_update: bool = False

    def __init__(
        self,
        n_classes: int,
        ignore_index: Optional[int] = None,
        absent_score: float = 0.0,
    ):
        """
        Args:
            n_classes: The number of classes.
            ignore_index: An optional class index to ignore in the computation.
            absent_score: The score to return for a class that is absent in both
                the prediction and target.
        """
        super().__init__()

        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.absent_score = absent_score

        # add_state registers tensors that track the metric's state.
        # 'dist_reduce_fx="sum"' ensures correct aggregation across devices.
        self.add_state('tp', default=torch.zeros(n_classes, dtype=torch.long), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.zeros(n_classes, dtype=torch.long), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.zeros(n_classes, dtype=torch.long), dist_reduce_fx='sum')
        self.add_state('support', default=torch.zeros(n_classes, dtype=torch.long), dist_reduce_fx='sum')

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update state with predictions and targets.

        Args:
            prediction: Predicted segmentation map, shape (B, H, W) or (B, C, H, W) with argmax.
            target: Ground truth segmentation map, shape (B, H, W).
        """
        # Using the functional API from torchmetrics is efficient.
        tp, fp, _, fn, support = multiclass_stat_scores(
            preds=prediction,
            target=target,
            num_classes=self.n_classes,
            ignore_index=self.ignore_index,
            average=None, # We want per-class stats
        )

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.support += support

    def compute(self) -> torch.Tensor:
        """
        Compute the final IoU score from the accumulated state.
        Returns a tensor of per-class IoU scores.
        """
        scores = torch.zeros(self.n_classes, device=self.tp.device, dtype=torch.float32)

        for i in range(self.n_classes):
            # Skip the ignored class index entirely.
            if i == self.ignore_index:
                continue

            # Denominator for the IoU calculation.
            denominator = self.tp[i] + self.fp[i] + self.fn[i]

            if denominator > 0:
                scores[i] = self.tp[i].float() / denominator
            # If the class is absent in both target (no support) and prediction (no tp or fp),
            # assign the absent_score.
            elif self.support[i] == 0:
                scores[i] = self.absent_score

        # If an ignore_index is set, we can return a masked or reduced tensor.
        # Here we return all scores and let the user handle the ignored index.
        return scores


# ====================================================================================


class PanopticMetric(Metric):
    """
    Computes Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition
    Quality (RQ) for panoptic segmentation, with optional temporal consistency checks.
    """
    full_state_update: bool = False

    def __init__(
        self,
        n_classes: int,
        temporally_consistent: bool = True,
        vehicles_id: int = 1,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.temporally_consistent = temporally_consistent
        self.vehicles_id = vehicles_id

        self.add_state('iou', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('true_positive', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('false_positive', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('false_negative', default=torch.zeros(n_classes), dist_reduce_fx='sum')

    def update(self, pred_instance: torch.Tensor, gt_instance: torch.Tensor) -> None:
        """
        Update state with predictions and targets.

        Args:
            pred_instance: (B, T, H, W) Temporally consistent instance prediction.
            gt_instance: (B, T, H, W) Ground truth instance segmentation.
        """
        batch_size, sequence_length = gt_instance.shape[:2]
        pred_segmentation = (pred_instance > 0).long()
        gt_segmentation = (gt_instance > 0).long()

        for b in range(batch_size):
            # Reset mapping for each sequence in the batch
            unique_id_mapping = {}
            for t in range(sequence_length):
                # Detach tensors from the computation graph before processing
                result = self._panoptic_metrics_step(
                    pred_segmentation[b, t].detach(),
                    pred_instance[b, t].detach(),
                    gt_segmentation[b, t],
                    gt_instance[b, t],
                    unique_id_mapping,
                )

                self.iou += result['iou']
                self.true_positive += result['true_positive']
                self.false_positive += result['false_positive']
                self.false_negative += result['false_negative']

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Computes the final Panoptic Quality (PQ), Segmentation Quality (SQ),
        and Recognition Quality (RQ).
        """
        # Clamp denominator to 1 to avoid division by zero.
        denominator = torch.clamp(self.true_positive + 0.5 * self.false_positive + 0.5 * self.false_negative, min=1.0)
        tp_clamp = torch.clamp(self.true_positive, min=1.0)

        pq = self.iou / denominator
        sq = self.iou / tp_clamp
        rq = self.true_positive / denominator

        return {'pq': pq, 'sq': sq, 'rq': rq}

    def _panoptic_metrics_step(self, pred_seg, pred_inst, gt_seg, gt_inst, unique_id_mapping) -> Dict[str, torch.Tensor]:
        """
        Helper to compute panoptic metrics for a single frame.
        This internal logic is highly specialized and remains unchanged.
        """
        # This implementation detail remains the same as the original
        keys = ['iou', 'true_positive', 'false_positive', 'false_negative']
        result = {key: torch.zeros(self.n_classes, dtype=torch.float32, device=gt_inst.device) for key in keys}

        n_instances = int(torch.cat([pred_inst, gt_inst]).max().item())
        n_all_things = n_instances + self.n_classes
        n_things_and_void = n_all_things + 1

        prediction, pred_to_cls = self._combine_mask(pred_seg, pred_inst, self.n_classes, n_all_things)
        target, target_to_cls = self._combine_mask(gt_seg, gt_inst, self.n_classes, n_all_things)

        x = prediction + n_things_and_void * target
        bincount_2d = torch.bincount(x.long(), minlength=n_things_and_void ** 2)
        conf = bincount_2d.reshape((n_things_and_void, n_things_and_void))[1:, 1:]

        union = conf.sum(0, keepdim=True) + conf.sum(1, keepdim=True) - conf
        iou = torch.where(union > 0, (conf.float()) / (union.float()), torch.zeros_like(union).float())

        mapping = (iou > 0.5).nonzero(as_tuple=False)
        is_matching = pred_to_cls[mapping[:, 1]] == target_to_cls[mapping[:, 0]]
        mapping = mapping[is_matching]
        tp_mask = torch.zeros_like(conf, dtype=torch.bool)
        tp_mask[mapping[:, 0], mapping[:, 1]] = True

        for target_id, pred_id in mapping:
            cls_id = pred_to_cls[pred_id]
            if self.temporally_consistent and cls_id == self.vehicles_id:
                if target_id.item() in unique_id_mapping and unique_id_mapping[target_id.item()] != pred_id.item():
                    result['false_negative'][target_to_cls[target_id]] += 1
                    result['false_positive'][pred_to_cls[pred_id]] += 1
                    unique_id_mapping[target_id.item()] = pred_id.item()
                    continue
            result['true_positive'][cls_id] += 1
            result['iou'][cls_id] += iou[target_id][pred_id]
            unique_id_mapping[target_id.item()] = pred_id.item()

        for target_id in range(self.n_classes, n_all_things):
            if not tp_mask[target_id, self.n_classes:].any() and target_to_cls[target_id] != -1:
                result['false_negative'][target_to_cls[target_id]] += 1
        for pred_id in range(self.n_classes, n_all_things):
            if not tp_mask[self.n_classes:, pred_id].any() and pred_to_cls[pred_id] != -1 and (conf[:, pred_id] > 0).any():
                result['false_positive'][pred_to_cls[pred_id]] += 1
        return result

    def _combine_mask(self, seg, inst, n_classes, n_all_things):
        """Helper to combine semantic and instance masks."""
        inst = inst.view(-1)
        instance_mask = inst > 0
        inst = inst - 1 + n_classes

        seg = seg.clone().view(-1)
        seg_mask = seg < n_classes

        id_to_cls_tuples = torch.cat((inst[instance_mask & seg_mask].unsqueeze(1), seg[instance_mask & seg_mask].unsqueeze(1)), dim=1)
        inst_id_to_cls = -id_to_cls_tuples.new_ones((n_all_things,))
        inst_id_to_cls[id_to_cls_tuples[:, 0]] = id_to_cls_tuples[:, 1]
        inst_id_to_cls[torch.arange(n_classes, device=seg.device)] = torch.arange(n_classes, device=seg.device)

        seg[instance_mask] = inst[instance_mask]
        seg += 1
        seg[~seg_mask] = 0
        return seg, inst_id_to_cls


# ====================================================================================


class PlanningMetric(Metric):
    """
    Evaluates motion planning performance based on collision with objects and
    L2 distance to the ground truth trajectory.
    """
    full_state_update: bool = False

    def __init__(self, cfg: Any, n_future: int = 4):
        super().__init__()
        # Use buffers for non-trainable parameters that should move with the model (e.g., to GPU)
        dx, bx, _ = gen_dx_bx(cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND)
        self.register_buffer("dx", dx[:2])
        self.register_buffer("bx", bx[:2])

        _, _, bev_dimension = calculate_birds_eye_view_parameters(cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND)
        self.bev_dimension = bev_dimension.numpy()

        self.W = cfg.EGO.WIDTH
        self.H = cfg.EGO.HEIGHT
        self.n_future = n_future

        self.add_state("obj_col", default=torch.zeros(n_future), dist_reduce_fx="sum")
        self.add_state("obj_box_col", default=torch.zeros(n_future), dist_reduce_fx="sum")
        self.add_state("l2", default=torch.zeros(n_future), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, trajs: torch.Tensor, gt_trajs: torch.Tensor, segmentation: torch.Tensor) -> None:
        """
        Update state with planned trajectories and ground truth.

        Args:
            trajs: Predicted trajectories (B, T, 3).
            gt_trajs: Ground truth trajectories (B, T, 3).
            segmentation: BEV semantic segmentation (B, T, H, W).
        """
        assert trajs.shape == gt_trajs.shape, "Prediction and ground truth shapes must match."
        l2_error = torch.sqrt(((trajs[..., :2] - gt_trajs[..., :2]) ** 2).sum(dim=-1))
        obj_coll, obj_box_coll = self._evaluate_collisions(trajs[..., :2], gt_trajs[..., :2], segmentation)

        self.obj_col += obj_coll
        self.obj_box_col += obj_box_coll
        self.l2 += l2_error.sum(dim=0)
        self.total += len(trajs)

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Compute final planning metrics, normalized by the number of samples.
        """
        if self.total == 0:
            return {'obj_col': torch.zeros_like(self.obj_col),
                    'obj_box_col': torch.zeros_like(self.obj_box_col),
                    'l2': torch.zeros_like(self.l2)}
        return {
            'obj_col': self.obj_col / self.total,
            'obj_box_col': self.obj_box_col / self.total,
            'l2': self.l2 / self.total,
        }

    def _evaluate_collisions(self, trajs, gt_trajs, segmentation):
        """Helper to evaluate collisions for a batch of trajectories."""
        B, T, _ = trajs.shape
        # This logic requires numpy, so we move data to CPU.
        # This can be a bottleneck, but is necessary for skimage.
        trajs = trajs.to('cpu').numpy() * np.array([-1, 1])
        gt_trajs = gt_trajs.to('cpu').numpy() * np.array([-1, 1])
        segmentation = segmentation.to('cpu').numpy()

        obj_coll_sum = np.zeros(T)
        obj_box_coll_sum = np.zeros(T)

        for i in range(B):
            gt_box_coll = self._evaluate_single_coll(gt_trajs[i], segmentation[i])
            pred_box_coll = self._evaluate_single_coll(trajs[i], segmentation[i])

            # Point collision
            yy, xx = trajs[i,:,0], trajs[i, :, 1]
            yi = ((yy - self.bx[0].cpu()) / self.dx[0].cpu()).long()
            xi = ((xx - self.bx[1].cpu()) / self.dx[1].cpu()).long()

            m1 = (yi >= 0) & (yi < self.bev_dimension[0]) & (xi >= 0) & (xi < self.bev_dimension[1]) & ~gt_box_coll
            ti = np.arange(T)
            obj_coll_sum[ti[m1]] += segmentation[i, ti[m1], yi[m1].numpy(), xi[m1].numpy()]

            # Box collision
            m2 = ~gt_box_coll
            obj_box_coll_sum[ti[m2]] += pred_box_coll[ti[m2]]

        device = self.obj_col.device
        return torch.from_numpy(obj_coll_sum).to(device), torch.from_numpy(obj_box_coll_sum).to(device)

    def _evaluate_single_coll(self, traj, segmentation):
        """Collision checking for a single trajectory using ego-vehicle footprint."""
        # Using numpy and scikit-image for polygon rasterization.
        pts = np.array([
            [-self.H / 2. + 0.5, self.W / 2.], [self.H / 2. + 0.5, self.W / 2.],
            [self.H / 2. + 0.5, -self.W / 2.], [-self.H / 2. + 0.5, -self.W / 2.],
        ])
        pts = (pts - self.bx.cpu().numpy()) / self.dx.cpu().numpy()
        pts[:, [0, 1]] = pts[:, [1, 0]]
        rr, cc = polygon(pts[:, 1], pts[:, 0])

        n_future, _ = traj.shape
        # Transform trajectory to BEV grid coordinates
        traj_grid = traj[:, ::-1] / self.dx.cpu().numpy()
        # Offset by ego footprint
        ego_offsets = np.stack([rr, cc], axis=-1)
        # Check collision at each future step
        collision = np.full(n_future, False)
        for t in range(n_future):
            future_pos = traj_grid[t] + ego_offsets
            r = np.clip(future_pos[:, 0].astype(np.int32), 0, self.bev_dimension[0] - 1)
            c = np.clip(future_pos[:, 1].astype(np.int32), 0, self.bev_dimension[1] - 1)
            collision[t] = np.any(segmentation[t, r, c])
        return collision