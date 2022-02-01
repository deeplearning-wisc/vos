import logging
import math
from typing import List
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn, distributions

# Detectron Imports
from detectron2.layers import ShapeSpec, cat
from detectron2.utils.events import get_event_storage

from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.retinanet import RetinaNet, RetinaNetHead, permute_to_N_HWA_K
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes

# Project Imports
from modeling.modeling_utils import covariance_output_to_cholesky, clamp_log_variance, get_probabilistic_loss_weight


@META_ARCH_REGISTRY.register()
class ProbabilisticRetinaNet(RetinaNet):
    """
    Probabilistic retinanet class.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # Parse configs
        self.cls_var_loss = cfg.MODEL.PROBABILISTIC_MODELING.CLS_VAR_LOSS.NAME
        self.compute_cls_var = self.cls_var_loss != 'none'
        self.cls_var_num_samples = cfg.MODEL.PROBABILISTIC_MODELING.CLS_VAR_LOSS.NUM_SAMPLES

        self.bbox_cov_loss = cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.NAME
        self.compute_bbox_cov = self.bbox_cov_loss != 'none'
        self.bbox_cov_num_samples = cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.NUM_SAMPLES

        self.bbox_cov_type = cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.COVARIANCE_TYPE
        if self.bbox_cov_type == 'diagonal':
            # Diagonal covariance matrix has N elements
            self.bbox_cov_dims = 4
        else:
            # Number of elements required to describe an NxN covariance matrix is
            # computed as:  (N * (N + 1)) / 2
            self.bbox_cov_dims = 10

        self.dropout_rate = cfg.MODEL.PROBABILISTIC_MODELING.DROPOUT_RATE
        self.use_dropout = self.dropout_rate != 0.0

        self.current_step = 0
        self.annealing_step = cfg.SOLVER.STEPS[1]

        # Define custom probabilistic head
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.head_in_features]
        self.head = ProbabilisticRetinaNetHead(
            cfg,
            self.use_dropout,
            self.dropout_rate,
            self.compute_cls_var,
            self.compute_bbox_cov,
            self.bbox_cov_dims,
            feature_shapes)

        # Send to device
        self.to(self.device)

    def forward(
            self,
            batched_inputs,
            return_anchorwise_output=False,
            num_mc_dropout_runs=-1):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

            return_anchorwise_output (bool): returns raw output for probabilistic inference

            num_mc_dropout_runs (int): perform efficient monte-carlo dropout runs by running only the head and
            not full neural network.

        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        # Preprocess image
        images = self.preprocess_image(batched_inputs)

        # Extract features and generate anchors
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]
        anchors = self.anchor_generator(features)

        # MC_Dropout inference forward
        if num_mc_dropout_runs > 1:
            anchors = anchors * num_mc_dropout_runs
            features = features * num_mc_dropout_runs
            output_dict = self.produce_raw_output(anchors, features)
            return output_dict

        # Regular inference forward
        if return_anchorwise_output:
            return self.produce_raw_output(anchors, features)

        # Training and validation forward
        pred_logits, pred_anchor_deltas, pred_logits_vars, pred_anchor_deltas_vars = self.head(
            features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [
            permute_to_N_HWA_K(
                x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [
            permute_to_N_HWA_K(
                x, 4) for x in pred_anchor_deltas]

        if pred_logits_vars is not None:
            pred_logits_vars = [
                permute_to_N_HWA_K(
                    x, self.num_classes) for x in pred_logits_vars]
        if pred_anchor_deltas_vars is not None:
            pred_anchor_deltas_vars = [permute_to_N_HWA_K(
                x, self.bbox_cov_dims) for x in pred_anchor_deltas_vars]

        if self.training:
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [
                x["instances"].to(
                    self.device) for x in batched_inputs]

            gt_classes, gt_boxes = self.label_anchors(
                anchors, gt_instances)

            self.anchors = torch.cat(
                [Boxes.cat(anchors).tensor for i in range(len(gt_instances))], 0)

            # Loss is computed based on what values are to be estimated by the neural
            # network
            losses = self.losses(
                anchors,
                gt_classes,
                gt_boxes,
                pred_logits,
                pred_anchor_deltas,
                pred_logits_vars,
                pred_anchor_deltas_vars)

            self.current_step += 1

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(
                        anchors, pred_logits, pred_anchor_deltas, images.image_sizes)
                    self.visualize_training(batched_inputs, results)
            return losses
        else:
            results = self.inference(
                anchors,
                pred_logits,
                pred_anchor_deltas,
                images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def losses(
            self,
            anchors,
            gt_classes,
            gt_boxes,
            pred_class_logits,
            pred_anchor_deltas,
            pred_class_logits_var=None,
            pred_bbox_cov=None):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits`, `pred_anchor_deltas`, `pred_class_logits_var` and `pred_bbox_cov`, see
                :meth:`RetinaNetHead.forward`.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        num_images = len(gt_classes)
        gt_labels = torch.stack(gt_classes)  # (N, R)
        anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)
        gt_anchor_deltas = [
            self.box2box_transform.get_deltas(
                anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + \
            (1 - self.loss_normalizer_momentum) * max(num_pos_anchors, 1)

        # classification and regression loss

        # Shapes:
        # (N x R, K) for class_logits and class_logits_var.
        # (N x R, 4), (N x R x 10) for pred_anchor_deltas and pred_class_bbox_cov respectively.

        # Transform per-feature layer lists to a single tensor
        pred_class_logits = cat(pred_class_logits, dim=1)
        pred_anchor_deltas = cat(pred_anchor_deltas, dim=1)

        if pred_class_logits_var is not None:
            pred_class_logits_var = cat(
                pred_class_logits_var, dim=1)

        if pred_bbox_cov is not None:
            pred_bbox_cov = cat(
                pred_bbox_cov, dim=1)

        gt_classes_target = torch.nn.functional.one_hot(
            gt_labels[valid_mask],
            num_classes=self.num_classes +
            1)[
            :,
            :-
            1].to(
            pred_class_logits[0].dtype)  # no loss for the last (background) class

        # Classification losses
        if self.compute_cls_var:
            # Compute classification variance according to:
            # "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NIPS 2017
            if self.cls_var_loss == 'loss_attenuation':
                num_samples = self.cls_var_num_samples
                # Compute standard deviation
                pred_class_logits_var = torch.sqrt(torch.exp(
                    pred_class_logits_var[valid_mask]))

                pred_class_logits = pred_class_logits[valid_mask]

                # Produce normal samples using logits as the mean and the standard deviation computed above
                # Scales with GPU memory. 12 GB ---> 3 Samples per anchor for
                # COCO dataset.
                univariate_normal_dists = distributions.normal.Normal(
                    pred_class_logits, scale=pred_class_logits_var)

                pred_class_stochastic_logits = univariate_normal_dists.rsample(
                    (num_samples,))
                pred_class_stochastic_logits = pred_class_stochastic_logits.view(
                    (pred_class_stochastic_logits.shape[1] * num_samples, pred_class_stochastic_logits.shape[2], -1))
                pred_class_stochastic_logits = pred_class_stochastic_logits.squeeze(
                    2)

                # Produce copies of the target classes to match the number of
                # stochastic samples.
                gt_classes_target = torch.unsqueeze(gt_classes_target, 0)
                gt_classes_target = torch.repeat_interleave(
                    gt_classes_target, num_samples, dim=0).view(
                    (gt_classes_target.shape[1] * num_samples, gt_classes_target.shape[2], -1))
                gt_classes_target = gt_classes_target.squeeze(2)

                # Produce copies of the target classes to form the stochastic
                # focal loss.
                loss_cls = sigmoid_focal_loss_jit(
                    pred_class_stochastic_logits,
                    gt_classes_target,
                    alpha=self.focal_loss_alpha,
                    gamma=self.focal_loss_gamma,
                    reduction="sum",
                ) / (num_samples * max(1, self.loss_normalizer))
            else:
                raise ValueError(
                    'Invalid classification loss name {}.'.format(
                        self.bbox_cov_loss))
        else:
            # Standard loss computation in case one wants to use this code
            # without any probabilistic inference.
            loss_cls = sigmoid_focal_loss_jit(
                pred_class_logits[valid_mask],
                gt_classes_target,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
            ) / max(1, self.loss_normalizer)

        # Compute Regression Loss
        pred_anchor_deltas = pred_anchor_deltas[pos_mask]
        gt_anchors_deltas = gt_anchor_deltas[pos_mask]
        if self.compute_bbox_cov:
            # We have to clamp the output variance else probabilistic metrics
            # go to infinity.
            pred_bbox_cov = clamp_log_variance(pred_bbox_cov[pos_mask])
            if self.bbox_cov_loss == 'negative_log_likelihood':
                if self.bbox_cov_type == 'diagonal':
                    # Compute regression variance according to:
                    # "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NIPS 2017
                    # This implementation with smooth_l1_loss outperforms using
                    # torch.distribution.multivariate_normal. Losses might have different numerical values
                    # since we do not include constants in this implementation.
                    loss_box_reg = 0.5 * torch.exp(-pred_bbox_cov) * smooth_l1_loss(
                        pred_anchor_deltas,
                        gt_anchors_deltas,
                        beta=self.smooth_l1_beta)
                    loss_covariance_regularize = 0.5 * pred_bbox_cov
                    loss_box_reg += loss_covariance_regularize

                    # Sum over all elements
                    loss_box_reg = torch.sum(
                        loss_box_reg) / max(1, self.loss_normalizer)
                else:
                    # Multivariate negative log likelihood. Implemented with
                    # pytorch multivariate_normal.log_prob function. Custom implementations fail to finish training
                    # due to NAN loss.

                    # This is the Cholesky decomposition of the covariance matrix. We reconstruct it from 10 estimated
                    # parameters as a lower triangular matrix.
                    forecaster_cholesky = covariance_output_to_cholesky(
                        pred_bbox_cov)

                    # Compute multivariate normal distribution using torch
                    # distribution functions.
                    multivariate_normal_dists = distributions.multivariate_normal.MultivariateNormal(
                        pred_anchor_deltas, scale_tril=forecaster_cholesky)

                    loss_box_reg = - \
                        multivariate_normal_dists.log_prob(gt_anchors_deltas)
                    loss_box_reg = torch.sum(
                        loss_box_reg) / max(1, self.loss_normalizer)

            elif self.bbox_cov_loss == 'second_moment_matching':
                # Compute regression covariance using second moment matching.
                loss_box_reg = smooth_l1_loss(
                    pred_anchor_deltas,
                    gt_anchors_deltas,
                    beta=self.smooth_l1_beta)

                # Compute errors
                errors = (pred_anchor_deltas - gt_anchors_deltas)

                if self.bbox_cov_type == 'diagonal':
                    # Compute second moment matching term.
                    second_moment_matching_term = smooth_l1_loss(
                        torch.exp(pred_bbox_cov), errors ** 2, beta=self.smooth_l1_beta)
                    loss_box_reg += second_moment_matching_term
                    loss_box_reg = torch.sum(
                        loss_box_reg) / max(1, self.loss_normalizer)
                else:
                    # Compute second moment matching term.
                    errors = torch.unsqueeze(errors, 2)
                    gt_error_covar = torch.matmul(
                        errors, torch.transpose(errors, 2, 1))

                    # This is the cholesky decomposition of the covariance matrix. We reconstruct it from 10 estimated
                    # parameters as a lower triangular matrix.
                    forecaster_cholesky = covariance_output_to_cholesky(
                        pred_bbox_cov)

                    predicted_covar = torch.matmul(
                        forecaster_cholesky, torch.transpose(
                            forecaster_cholesky, 2, 1))

                    second_moment_matching_term = smooth_l1_loss(
                        predicted_covar, gt_error_covar, beta=self.smooth_l1_beta, reduction='sum')

                    loss_box_reg = (torch.sum(
                        loss_box_reg) + second_moment_matching_term) / max(1, self.loss_normalizer)

            elif self.bbox_cov_loss == 'energy_loss':
                # Compute regression variance according to energy score loss.
                forecaster_means = pred_anchor_deltas

                # Compute forecaster cholesky. Takes care of diagonal case
                # automatically.
                forecaster_cholesky = covariance_output_to_cholesky(
                    pred_bbox_cov)

                # Define normal distribution samples. To compute energy score,
                # we need i+1 samples.

                # Define per-anchor Distributions
                multivariate_normal_dists = distributions.multivariate_normal.MultivariateNormal(
                    forecaster_means, scale_tril=forecaster_cholesky)

                # Define Monte-Carlo Samples
                distributions_samples = multivariate_normal_dists.rsample(
                    (self.bbox_cov_num_samples + 1,))

                distributions_samples_1 = distributions_samples[0:self.bbox_cov_num_samples, :, :]
                distributions_samples_2 = distributions_samples[1:
                                                                self.bbox_cov_num_samples + 1, :, :]

                # Compute energy score
                gt_anchors_deltas_samples = torch.repeat_interleave(
                    gt_anchors_deltas.unsqueeze(0), self.bbox_cov_num_samples, dim=0)

                energy_score_first_term = 2.0 * smooth_l1_loss(
                    distributions_samples_1,
                    gt_anchors_deltas_samples,
                    beta=self.smooth_l1_beta,
                    reduction="sum") / self.bbox_cov_num_samples  # First term

                energy_score_second_term = - smooth_l1_loss(
                    distributions_samples_1,
                    distributions_samples_2,
                    beta=self.smooth_l1_beta,
                    reduction="sum") / self.bbox_cov_num_samples   # Second term

                # Final Loss
                loss_box_reg = (
                    energy_score_first_term + energy_score_second_term) / max(1, self.loss_normalizer)

            else:
                raise ValueError(
                    'Invalid regression loss name {}.'.format(
                        self.bbox_cov_loss))

            # Perform loss annealing. Essential for reliably training variance estimates using NLL in RetinaNet.
            # For energy score and second moment matching, this is optional.
            standard_regression_loss = smooth_l1_loss(
                pred_anchor_deltas,
                gt_anchors_deltas,
                beta=self.smooth_l1_beta,
                reduction="sum",
            ) / max(1, self.loss_normalizer)

            probabilistic_loss_weight = get_probabilistic_loss_weight(
                self.current_step, self.annealing_step)
            loss_box_reg = (1.0 - probabilistic_loss_weight) * \
                standard_regression_loss + probabilistic_loss_weight * loss_box_reg
        else:
            # Standard regression loss in case no variance is needed to be
            # estimated.
            loss_box_reg = smooth_l1_loss(
                pred_anchor_deltas,
                gt_anchors_deltas,
                beta=self.smooth_l1_beta,
                reduction="sum",
            ) / max(1, self.loss_normalizer)

        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}

    def produce_raw_output(self, anchors, features):
        """
        Given anchors and features, produces raw pre-nms output to be used for custom fusion operations.
        """
        # Perform inference run
        pred_logits, pred_anchor_deltas, pred_logits_vars, pred_anchor_deltas_vars = self.head(
            features)

        # Transpose the Hi*Wi*A dimension to the middle:
        pred_logits = [
            permute_to_N_HWA_K(
                x, self.num_classes) for x in pred_logits]
        pred_anchor_deltas = [
            permute_to_N_HWA_K(
                x, 4) for x in pred_anchor_deltas]

        if pred_logits_vars is not None:
            pred_logits_vars = [
                permute_to_N_HWA_K(
                    x, self.num_classes) for x in pred_logits_vars]
        if pred_anchor_deltas_vars is not None:
            pred_anchor_deltas_vars = [permute_to_N_HWA_K(
                x, self.bbox_cov_dims) for x in pred_anchor_deltas_vars]

        # Create raw output dictionary
        raw_output = {'anchors': anchors}

        # Shapes:
        # (N x R, K) for class_logits and class_logits_var.
        # (N x R, 4), (N x R x 10) for pred_anchor_deltas and pred_class_bbox_cov respectively.
        raw_output.update({'box_cls': pred_logits,
                           'box_delta': pred_anchor_deltas,
                           'box_cls_var': pred_logits_vars,
                           'box_reg_var': pred_anchor_deltas_vars})
        return raw_output


class ProbabilisticRetinaNetHead(RetinaNetHead):
    """
    The head used in ProbabilisticRetinaNet for object class probability estimation, box regression, box covariance estimation.
    It has three subnets for the three tasks, with a common structure but separate parameters.
    """

    def __init__(self,
                 cfg,
                 use_dropout,
                 dropout_rate,
                 compute_cls_var,
                 compute_bbox_cov,
                 bbox_cov_dims,
                 input_shape: List[ShapeSpec]):
        super().__init__(cfg, input_shape)

        # Extract config information
        # fmt: off
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
        num_convs = cfg.MODEL.RETINANET.NUM_CONVS
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        num_anchors = build_anchor_generator(cfg, input_shape).num_cell_anchors
        # fmt: on
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        self.compute_cls_var = compute_cls_var
        self.compute_bbox_cov = compute_bbox_cov
        self.bbox_cov_dims = bbox_cov_dims

        # For consistency all configs are grabbed from original RetinaNet
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        cls_subnet = []
        bbox_subnet = []
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            cls_subnet.append(nn.ReLU())

            bbox_subnet.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            bbox_subnet.append(nn.ReLU())

            if self.use_dropout:
                cls_subnet.append(nn.Dropout(p=self.dropout_rate))
                bbox_subnet.append(nn.Dropout(p=self.dropout_rate))

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)

        self.cls_score = nn.Conv2d(
            in_channels,
            num_anchors *
            num_classes,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bbox_pred = nn.Conv2d(
            in_channels,
            num_anchors * 4,
            kernel_size=3,
            stride=1,
            padding=1)

        for modules in [
                self.cls_subnet,
                self.bbox_subnet,
                self.cls_score,
                self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

        # Create subnet for classification variance estimation.
        if self.compute_cls_var:
            self.cls_var = nn.Conv2d(
                in_channels,
                num_anchors *
                num_classes,
                kernel_size=3,
                stride=1,
                padding=1)

            for layer in self.cls_var.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, -10.0)

        # Create subnet for bounding box covariance estimation.
        if self.compute_bbox_cov:
            self.bbox_cov = nn.Conv2d(
                in_channels,
                num_anchors * self.bbox_cov_dims,
                kernel_size=3,
                stride=1,
                padding=1)

            for layer in self.bbox_cov.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.0001)
                    torch.nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the classification probability
                at each spatial position for each of the A anchors and K object
                classes.
            logits_var (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
                The tensor predicts the variance of the logits modeled as a univariate
                Gaussian distribution at each spatial position for each of the A anchors and K object
                classes.

            bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
                The tensor predicts 4-vector (dx,dy,dw,dh) box
                regression values for every anchor. These values are the
                relative offset between the anchor and the ground truth box.

            bbox_cov (list[Tensor]): #lvl tensors, each has shape (N, Ax4 or Ax10, Hi, Wi).
                The tensor predicts elements of the box
                covariance values for every anchor. The dimensions of the box covarianc
                depends on estimating a full covariance (10) or a diagonal covariance matrix (4).
        """
        logits = []
        bbox_reg = []

        logits_var = []
        bbox_cov = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
            if self.compute_cls_var:
                logits_var.append(self.cls_var(self.cls_subnet(feature)))
            if self.compute_bbox_cov:
                bbox_cov.append(self.bbox_cov(self.bbox_subnet(feature)))

        return_vector = [logits, bbox_reg]

        if self.compute_cls_var:
            return_vector.append(logits_var)
        else:
            return_vector.append(None)

        if self.compute_bbox_cov:
            return_vector.append(bbox_cov)
        else:
            return_vector.append(None)

        return return_vector
