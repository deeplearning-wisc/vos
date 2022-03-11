import numpy as np
import os
import torch

from PIL import Image

# Detectron imports
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.layers import batched_nms
from detectron2.structures import BoxMode, Boxes, Instances, pairwise_iou

# Project imports
from inference.image_corruptions import corruption_dict, corruption_tuple
from inference.rcnn_predictor import GeneralizedRcnnPlainPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_predictor(cfg):
    """
    Builds probabilistic predictor according to architecture in config file.
    Args:
        cfg (CfgNode): detectron2 configuration node.

    Returns:
        Instance of the correct predictor.
    """
    if cfg.MODEL.META_ARCHITECTURE == 'ProbabilisticRetinaNet':
        return RetinaNetProbabilisticPredictor(cfg)
    elif cfg.MODEL.META_ARCHITECTURE == 'ProbabilisticGeneralizedRCNN':
        return GeneralizedRcnnProbabilisticPredictor(cfg)
    elif cfg.MODEL.META_ARCHITECTURE == 'ProbabilisticDetr':
        return DetrProbabilisticPredictor(cfg)
    elif cfg.MODEL.META_ARCHITECTURE == "GeneralizedRCNN":
        return GeneralizedRcnnPlainPredictor(cfg)
    elif cfg.MODEL.META_ARCHITECTURE == 'GeneralizedRCNNLogisticGMM':
        return GeneralizedRcnnPlainPredictor(cfg)
    else:
        raise ValueError(
            'Invalid meta-architecture {}.'.format(cfg.MODEL.META_ARCHITECTURE))


def general_standard_nms_postprocessing(input_im,
                                        outputs,
                                        nms_threshold=0.5,
                                        max_detections_per_image=100):
    """

    Args:
        input_im (list): an input im list generated from dataset handler.
        outputs (list): output list form model specific inference function
        nms_threshold (float): non-maximum suppression threshold
        max_detections_per_image (int): maximum allowed number of detections per image.

    Returns:
        result (Instances): final results after nms

    """
    logistic_score = None
    try:
        predicted_boxes, predicted_boxes_covariance, predicted_prob, inter_feat, \
        classes_idxs, predicted_prob_vectors, det_labels = outputs
    except:
        predicted_boxes, predicted_boxes_covariance, predicted_prob, inter_feat, logistic_score, \
        classes_idxs, predicted_prob_vectors, det_labels = outputs

    # Perform nms
    keep = batched_nms(
        predicted_boxes,
        predicted_prob,
        classes_idxs,
        nms_threshold)
    keep = keep[: max_detections_per_image]
    # import ipdb; ipdb.set_trace()
    # Keep highest scoring results
    result = Instances(
        (input_im[0]['image'].shape[1],
         input_im[0]['image'].shape[2]))
    result.pred_boxes = Boxes(predicted_boxes[keep])
    result.scores = predicted_prob[keep]
    result.pred_classes = classes_idxs[keep]
    result.pred_cls_probs = predicted_prob_vectors[keep]
    result.inter_feat = inter_feat[keep]
    result.det_labels = det_labels[keep]
    if logistic_score is not None:
        result.logistic_score = logistic_score[keep]

    # Handle case where there is no covariance matrix such as classical
    # inference.
    if isinstance(predicted_boxes_covariance, torch.Tensor):
        result.pred_boxes_covariance = predicted_boxes_covariance[keep]
    else:
        result.pred_boxes_covariance = torch.zeros(
            predicted_boxes[keep].shape + (4,)).to(device)

    return result


def general_output_statistics_postprocessing(input_im,
                                             outputs,
                                             nms_threshold=0.5,
                                             max_detections_per_image=100,
                                             affinity_threshold=0.7):
    """

    Args:
        input_im (list): an input im list generated from dataset handler.
        outputs (list): output list form model specific inference function
        nms_threshold (float): non-maximum suppression threshold between 0-1
        max_detections_per_image (int): maximum allowed number of detections per image.
        affinity_threshold (float): cluster affinity threshold between 0-1
    Returns:
        result (Instances): final results after nms

    """

    predicted_boxes, predicted_boxes_covariance, predicted_prob, classes_idxs, predicted_prob_vectors = outputs

    # Get pairwise iou matrix
    match_quality_matrix = pairwise_iou(
        Boxes(predicted_boxes), Boxes(predicted_boxes))

    # Get cluster centers using standard nms. Much faster than sequential
    # clustering.
    keep = batched_nms(
        predicted_boxes,
        predicted_prob,
        classes_idxs,
        nms_threshold)

    keep = keep[: max_detections_per_image]

    clusters_inds = match_quality_matrix[keep, :]
    clusters_inds = clusters_inds > affinity_threshold

    # Compute mean and covariance for every cluster.
    predicted_prob_vectors_list = []
    predicted_boxes_list = []
    predicted_boxes_covariance_list = []

    for cluster_idxs, center_idx in zip(
            clusters_inds, keep):

        if cluster_idxs.sum(0) >= 2:
            # Make sure to only select cluster members of same class as center
            cluster_center_classes_idx = classes_idxs[center_idx]
            cluster_classes_idxs = classes_idxs[cluster_idxs]
            class_similarity_idxs = cluster_classes_idxs == cluster_center_classes_idx

            # Grab cluster
            box_cluster = predicted_boxes[cluster_idxs,
                                          :][class_similarity_idxs, :]

            cluster_mean = box_cluster.mean(0)

            residuals = (box_cluster - cluster_mean).unsqueeze(2)
            cluster_covariance = torch.sum(torch.matmul(residuals, torch.transpose(
                residuals, 2, 1)), 0) / max((box_cluster.shape[0] - 1), 1.0)

            # Assume final result as mean and covariance of gaussian mixture of cluster members if
            # covariance is provided by neural network.
            if predicted_boxes_covariance is not None:
                if len(predicted_boxes_covariance) > 0:
                    cluster_covariance = cluster_covariance + \
                        predicted_boxes_covariance[cluster_idxs, :][class_similarity_idxs, :].mean(0)

            # Compute average over cluster probabilities
            cluster_probs_vector = predicted_prob_vectors[cluster_idxs, :][class_similarity_idxs, :].mean(
                0)
        else:
            cluster_mean = predicted_boxes[center_idx]
            cluster_probs_vector = predicted_prob_vectors[center_idx]
            cluster_covariance = 1e-4 * torch.eye(4, 4).to(device)
            if predicted_boxes_covariance is not None:
                if len(predicted_boxes_covariance) > 0:
                    cluster_covariance = predicted_boxes_covariance[center_idx]

        predicted_boxes_list.append(cluster_mean)
        predicted_boxes_covariance_list.append(cluster_covariance)
        predicted_prob_vectors_list.append(cluster_probs_vector)

    result = Instances(
        (input_im[0]['image'].shape[1],
         input_im[0]['image'].shape[2]))

    if len(predicted_boxes_list) > 0:
        # We do not average the probability vectors for this post processing method. Averaging results in
        # very low mAP due to mixing with low scoring detection instances.
        result.pred_boxes = Boxes(torch.stack(predicted_boxes_list, 0))
        predicted_prob_vectors = torch.stack(predicted_prob_vectors_list, 0)
        predicted_prob, classes_idxs = torch.max(
            predicted_prob_vectors, 1)
        result.scores = predicted_prob
        result.pred_classes = classes_idxs
        result.pred_cls_probs = predicted_prob_vectors
        result.pred_boxes_covariance = torch.stack(
            predicted_boxes_covariance_list, 0)
    else:
        result.pred_boxes = Boxes(predicted_boxes)
        result.scores = torch.zeros(predicted_boxes.shape[0]).to(device)
        result.pred_classes = classes_idxs
        result.pred_cls_probs = predicted_prob_vectors
        result.pred_boxes_covariance = torch.empty(
            (predicted_boxes.shape + (4,))).to(device)
    return result


def general_black_box_ensembles_post_processing(
        input_im,
        ensemble_pred_box_list,
        ensembles_class_idxs_list,
        ensemble_pred_prob_vectors_list,
        ensembles_pred_box_covariance_list,
        nms_threshold=0.5,
        max_detections_per_image=100,
        affinity_threshold=0.7,
        is_generalized_rcnn=False,
        merging_method='mixture_of_gaussians'):
    """

    Args:
        input_im (list): an input im list generated from dataset handler.
        ensemble_pred_box_list (list): predicted box list
        ensembles_class_idxs_list (list): predicted classes list
        ensemble_pred_prob_vectors_list (list): predicted probability vector list
        ensembles_pred_box_covariance_list (list): predicted covariance matrices
        nms_threshold (float): non-maximum suppression threshold between 0-1
        max_detections_per_image (int): Number of maximum allowable detections per image.
        affinity_threshold (float): cluster affinity threshold between 0-1
        is_generalized_rcnn (bool): used to handle category selection by removing background class.
        merging_method (str): default is gaussian mixture model. use 'bayesian_inference' to perform gaussian inference
        similar to bayesod.
    Returns:
        result (Instances): final results after nms

    """

    predicted_boxes = torch.cat(ensemble_pred_box_list, 0)
    predicted_boxes_covariance = torch.cat(
        ensembles_pred_box_covariance_list, 0)
    predicted_prob_vectors = torch.cat(
        ensemble_pred_prob_vectors_list, 0)
    predicted_class_idxs = torch.cat(ensembles_class_idxs_list, 0)

    # Compute iou between all output boxes and each other output box.
    match_quality_matrix = pairwise_iou(
        Boxes(predicted_boxes), Boxes(predicted_boxes))

    # Perform basic sequential clustering.
    clusters = []
    for i in range(match_quality_matrix.shape[0]):
        # Check if current box is already a member of any previous cluster.
        if i != 0:
            all_clusters = torch.cat(clusters, 0)
            if (all_clusters == i).any():
                continue
        # Only add if boxes have the same category.
        cluster_membership_test = (match_quality_matrix[i,
                                                        :] >= affinity_threshold) & (
            predicted_class_idxs == predicted_class_idxs[i])
        inds = torch.where(cluster_membership_test)
        clusters.extend(inds)

    # Compute mean and covariance for every cluster.
    predicted_boxes_list = []
    predicted_boxes_covariance_list = []
    predicted_prob_vectors_list = []

    # Compute cluster mean and covariance matrices.
    for cluster in clusters:
        box_cluster = predicted_boxes[cluster]
        box_cluster_covariance = predicted_boxes_covariance[cluster]
        if box_cluster.shape[0] >= 2:
            if merging_method == 'mixture_of_gaussians':
                cluster_mean = box_cluster.mean(0)

                # Compute epistemic covariance
                residuals = (box_cluster - cluster_mean).unsqueeze(2)
                predicted_covariance = torch.sum(torch.matmul(residuals, torch.transpose(
                    residuals, 2, 1)), 0) / (box_cluster.shape[0] - 1)

                # Add epistemic covariance
                predicted_covariance = predicted_covariance + \
                    box_cluster_covariance.mean(0)

                predicted_boxes_list.append(cluster_mean)
                predicted_boxes_covariance_list.append(predicted_covariance)
                predicted_prob_vectors_list.append(
                    predicted_prob_vectors[cluster].mean(0))
            else:
                cluster_mean, predicted_covariance = bounding_box_bayesian_inference(box_cluster.cpu(
                ).numpy(), box_cluster_covariance.cpu().numpy(), box_merge_mode='bayesian_inference')
                cluster_mean = torch.as_tensor(cluster_mean).to(device)
                predicted_covariance = torch.as_tensor(
                    predicted_covariance).to(device)

                predicted_boxes_list.append(cluster_mean)
                predicted_boxes_covariance_list.append(predicted_covariance)
                predicted_prob_vectors_list.append(
                    predicted_prob_vectors[cluster].mean(0))
        else:
            predicted_boxes_list.append(predicted_boxes[cluster].mean(0))
            predicted_boxes_covariance_list.append(
                predicted_boxes_covariance[cluster].mean(0))
            predicted_prob_vectors_list.append(
                predicted_prob_vectors[cluster].mean(0))

    result = Instances(
        (input_im[0]['image'].shape[1],
         input_im[0]['image'].shape[2]))

    if len(predicted_boxes_list) > 0:
        predicted_prob_vectors = torch.stack(predicted_prob_vectors_list, 0)

        # Remove background class if generalized rcnn
        if is_generalized_rcnn:
            predicted_prob_vectors_no_bkg = predicted_prob_vectors[:, :-1]
        else:
            predicted_prob_vectors_no_bkg = predicted_prob_vectors

        predicted_prob, classes_idxs = torch.max(
            predicted_prob_vectors_no_bkg, 1)
        predicted_boxes = torch.stack(predicted_boxes_list, 0)

        # We want to keep the maximum allowed boxes per image to be consistent
        # with the rest of the methods. However, just sorting by score or uncertainty will lead to a lot of
        # redundant detections so we have to use one more NMS step.
        keep = batched_nms(
            predicted_boxes,
            predicted_prob,
            classes_idxs,
            nms_threshold)
        keep = keep[:max_detections_per_image]

        result.pred_boxes = Boxes(predicted_boxes[keep])
        result.scores = predicted_prob[keep]
        result.pred_classes = classes_idxs[keep]
        result.pred_cls_probs = predicted_prob_vectors[keep]
        result.pred_boxes_covariance = torch.stack(
            predicted_boxes_covariance_list, 0)[keep]
    else:
        result.pred_boxes = Boxes(predicted_boxes)
        result.scores = torch.zeros(predicted_boxes.shape[0]).to(device)
        result.pred_classes = predicted_class_idxs
        result.pred_cls_probs = predicted_prob_vectors
        result.pred_boxes_covariance = torch.empty(
            (predicted_boxes.shape + (4,))).to(device)
    return result


def bounding_box_bayesian_inference(cluster_means,
                                    cluster_covs,
                                    box_merge_mode):
    """

    Args:
        cluster_means (nd array): cluster box means.
        cluster_covs (nd array): cluster box covariance matrices.
        box_merge_mode (str): whether to use covariance intersection or not
    Returns:
        final_mean (nd array): cluster fused mean.
        final_cov (nd array): cluster fused covariance matrix.
    """

    cluster_precs = np.linalg.inv(cluster_covs)
    if box_merge_mode == 'bayesian_inference':
        final_cov = np.linalg.inv(cluster_precs.sum(0))

        final_mean = np.matmul(
            cluster_precs, np.expand_dims(cluster_means, 2)).sum(0)
        final_mean = np.squeeze(np.matmul(final_cov, final_mean))
    elif box_merge_mode == 'covariance_intersection':
        cluster_difference_precs = cluster_precs.sum(0) - cluster_precs

        cluster_precs_det = np.linalg.det(cluster_precs)
        cluster_total_prec_det = np.linalg.det(cluster_precs.sum(0))
        cluster_difference_precs_det = np.linalg.det(
            cluster_difference_precs)
        omegas = (cluster_total_prec_det - cluster_difference_precs_det + cluster_precs_det) / (
            cluster_precs.shape[0] * cluster_total_prec_det +
            (cluster_precs_det - cluster_difference_precs_det).sum(0))

        weighted_cluster_precs = np.expand_dims(
            omegas, (1, 2)) * cluster_precs
        final_cov = np.linalg.inv(weighted_cluster_precs.sum(0))

        final_mean =  np.squeeze(np.matmul(
            final_cov,
            np.matmul(
                weighted_cluster_precs,
                np.expand_dims(cluster_means, 2)).sum(0)))

    return final_mean, final_cov


def compute_mean_covariance_torch(input_samples):
    """
    Function for efficient computation of mean and covariance matrix in pytorch.

    Args:
        input_samples(list): list of tensors from M stochastic monte-carlo sampling runs, each containing N x k tensors.

    Returns:
        predicted_mean(Tensor): an Nxk tensor containing the predicted mean.
        predicted_covariance(Tensor): an Nxkxk tensor containing the predicted covariance matrix.

    """
    if isinstance(input_samples, torch.Tensor):
        num_samples = input_samples.shape[2]
    else:
        num_samples = len(input_samples)
        input_samples = torch.stack(input_samples, 2)

    # Compute Mean
    predicted_mean = torch.mean(input_samples, 2, keepdim=True)

    # Compute Covariance
    residuals = torch.transpose(
        torch.unsqueeze(
            input_samples -
            predicted_mean,
            1),
        1,
        3)
    predicted_covariance = torch.matmul(
        residuals, torch.transpose(residuals, 3, 2))
    predicted_covariance = torch.sum(
        predicted_covariance, 1) / (num_samples - 1)

    return predicted_mean.squeeze(2), predicted_covariance


def probabilistic_detector_postprocess(
        results,
        output_height,
        output_width):
    """
    Resize the output instances and scales estimated covariance matrices.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    Args:
        results (Dict): the raw outputs from the probabilistic detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height: the desired output resolution.
        output_width: the desired output resolution.

    Returns:
        results (Dict): dictionary updated with rescaled boxes and covariance matrices.
    """
    scale_x, scale_y = (output_width /
                        results.image_size[1], output_height /
                        results.image_size[0])
    results = Instances((output_height, output_width), **results.get_fields())

    output_boxes = results.pred_boxes

    # Scale bounding boxes
    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)
    results = results[output_boxes.nonempty()]

    # Scale covariance matrices
    if results.has("pred_boxes_covariance"):
        # Add small value to make sure covariance matrix is well conditioned
        output_boxes_covariance = results.pred_boxes_covariance + 1e-4 * \
            torch.eye(results.pred_boxes_covariance.shape[2]).to(device)

        scale_mat = torch.diag_embed(
            torch.as_tensor(
                (scale_x,
                 scale_y,
                 scale_x,
                 scale_y))).to(device).unsqueeze(0)
        scale_mat = torch.repeat_interleave(
            scale_mat, output_boxes_covariance.shape[0], 0)
        output_boxes_covariance = torch.matmul(
            torch.matmul(
                scale_mat,
                output_boxes_covariance),
            torch.transpose(scale_mat, 2, 1))
        results.pred_boxes_covariance = output_boxes_covariance
    return results


def covar_xyxy_to_xywh(output_boxes_covariance):
    """
    Converts covariance matrices from top-left bottom-right corner representation to top-left corner
    and width-height representation.

    Args:
        output_boxes_covariance: Input covariance matrices.

    Returns:
        output_boxes_covariance (Nxkxk): Transformed covariance matrices
    """
    transformation_mat = torch.as_tensor([[1.0, 0, 0, 0],
                                          [0, 1.0, 0, 0],
                                          [-1.0, 0, 1.0, 0],
                                          [0, -1.0, 0, 1.0]]).to(device).unsqueeze(0)
    transformation_mat = torch.repeat_interleave(
        transformation_mat, output_boxes_covariance.shape[0], 0)
    output_boxes_covariance = torch.matmul(
        torch.matmul(
            transformation_mat,
            output_boxes_covariance),
        torch.transpose(transformation_mat, 2, 1))

    return output_boxes_covariance


def instances_to_json(instances, img_id, cat_mapping_dict=None):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances): detectron2 instances
        img_id (int): the image id
        cat_mapping_dict (dict): dictionary to map between raw category id from net and dataset id. very important if
        performing inference on different dataset than that used for training.

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.cpu().numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.cpu().tolist()
    classes = instances.pred_classes.cpu().tolist()
    inter_feat = instances.inter_feat.cpu().tolist()
    if instances.has('logistic_score'):
        logistic_score = instances.logistic_score.cpu().tolist()
    # import ipdb; ipdb.set_trace()

    classes = [
        cat_mapping_dict[class_i] if class_i in cat_mapping_dict.keys() else -
        1 for class_i in classes]
    # breakpoint()
    pred_cls_probs = instances.pred_cls_probs.cpu().tolist()

    if instances.has("pred_boxes_covariance"):
        pred_boxes_covariance = covar_xyxy_to_xywh(
            instances.pred_boxes_covariance).cpu().tolist()
    else:
        pred_boxes_covariance = []

    results = []
    for k in range(num_instance):
        if classes[k] != -1:
            if instances.has('logistic_score'):
                result = {
                    "image_id": img_id,
                    "category_id": classes[k],
                    "bbox": boxes[k],
                    "score": scores[k],
                    "inter_feat": inter_feat[k],
                    "logistic_score": logistic_score[k],
                    "cls_prob": pred_cls_probs[k],
                    "bbox_covar": pred_boxes_covariance[k]
                }
            else:
                result = {
                    "image_id": img_id,
                    "category_id": classes[k],
                    "bbox": boxes[k],
                    "score": scores[k],
                    "inter_feat": inter_feat[k],
                    "cls_prob": pred_cls_probs[k],
                    "bbox_covar": pred_boxes_covariance[k]
                }

            results.append(result)
    return results


class SampleBox2BoxTransform(Box2BoxTransform):
    """
    Extension of Box2BoxTransform to support transforming across batch sizes.
    """

    def apply_samples_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2, :] - boxes[:, 0, :]
        heights = boxes[:, 3, :] - boxes[:, 1, :]
        ctr_x = boxes[:, 0, :] + 0.5 * widths
        ctr_y = boxes[:, 1, :] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = deltas[:, 0::4, :] / wx
        dy = deltas[:, 1::4, :] / wy
        dw = deltas[:, 2::4, :] / ww
        dh = deltas[:, 3::4, :] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4, :] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4, :] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4, :] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4, :] = pred_ctr_y + 0.5 * pred_h  # y2
        return pred_boxes


def corrupt(x, severity=1, corruption_name=None, corruption_number=None):
    """
    :param x: image to corrupt; a 224x224x3 numpy array in [0, 255]
    :param severity: strength with which to corrupt x; an integer in [0, 5]
    :param corruption_name: specifies which corruption function to call;
    must be one of 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate';
                    the last four are validation functions
    :param corruption_number: the position of the corruption_name in the above list;
    an integer in [0, 18]; useful for easy looping; 15, 16, 17, 18 are validation corruption numbers
    :return: the image x corrupted by a corruption function at the given severity; same shape as input
    """

    if corruption_name is not None:
        x_corrupted = corruption_dict[corruption_name](
            Image.fromarray(x), severity)
    elif corruption_number is not None:
        x_corrupted = corruption_tuple[corruption_number](
            Image.fromarray(x), severity)
    else:
        raise ValueError(
            "Either corruption_name or corruption_number must be passed")

    if x_corrupted.shape != x.shape:
        raise AssertionError("Output image not same size as input image!")

    return np.uint8(x_corrupted)


def get_dir_alphas(pred_class_logits):
    """
    Function to get dirichlet parameters from logits
    Args:
        pred_class_logits: class logits
    """
    return torch.relu_(pred_class_logits) + 1.0


def get_inference_output_dir(output_dir_name,
                             test_dataset_name,
                             inference_config_name,
                             image_corruption_level):
    return os.path.join(
        output_dir_name,
        'inference',
        test_dataset_name,
        os.path.split(inference_config_name)[-1][:-5],
        "corruption_level_" + str(image_corruption_level))
