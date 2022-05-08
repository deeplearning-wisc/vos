import numpy as np
import torch


# Detectron Imports
from detectron2.layers import batched_nms
from detectron2.structures import Boxes, Instances, pairwise_iou

# Project Imports
from inference import inference_utils
from inference.inference_core import ProbabilisticPredictor
from modeling.modeling_utils import covariance_output_to_cholesky, clamp_log_variance


class GeneralizedRcnnPlainPredictor(ProbabilisticPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Define test score threshold
        self.test_score_thres = self.model.roi_heads.box_predictor.test_score_thresh
        self.test_nms_thresh = self.model.roi_heads.box_predictor.test_nms_thresh
        self.test_topk_per_image = self.model.roi_heads.box_predictor.test_topk_per_image

        # Create transform
        self.sample_box2box_transform = inference_utils.SampleBox2BoxTransform(
            self.cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

        # Put proposal generator in eval mode if dropout enabled
        if self.mc_dropout_enabled:
            self.model.proposal_generator.eval()

    def generalized_rcnn_probabilistic_inference(self,
                                                 input_im,
                                                 outputs=None,
                                                 ensemble_inference=False,
                                                 outputs_list=None):
        """
        General RetinaNet probabilistic anchor-wise inference. Preliminary inference step for many post-processing
        based inference methods such as standard_nms, output_statistics, and bayes_od.
        Args:
            input_im (list): an input im list generated from dataset handler.
            outputs (list): outputs from model.forward(). will be computed internally if not provided.
            ensemble_inference (bool): True if ensembles are used for inference. If set to true, outputs_list must be externally provided.
            outputs_list (list): List of model() outputs, usually generated from ensembles of models.
        Returns:
            all_predicted_boxes,
            all_predicted_boxes_covariance (Tensor): Nx4x4 vectors used
            all_predicted_prob (Tensor): Nx1 scores which represent max of all_pred_prob_vectors. For usage in NMS and mAP computation.
            all_classes_idxs (Tensor): Nx1 Class ids to be used for NMS.
            all_predicted_prob_vectors (Tensor): NxK tensor where K is the number of classes.
        """
        is_epistemic = ((self.mc_dropout_enabled and self.num_mc_dropout_runs > 1)
                        or ensemble_inference) and outputs is None
        if is_epistemic:
            if self.mc_dropout_enabled and self.num_mc_dropout_runs > 1:
                outputs_list = self.model(
                    input_im,
                    return_anchorwise_output=True,
                    num_mc_dropout_runs=self.num_mc_dropout_runs)

            proposals_list = [outputs['proposals']
                              for outputs in outputs_list]
            box_delta_list = [outputs['box_delta']
                              for outputs in outputs_list]
            box_cls_list = [outputs['box_cls'] for outputs in outputs_list]
            box_reg_var_list = [outputs['box_reg_var']
                                for outputs in outputs_list]
            box_cls_var_list = [outputs['box_cls_var']
                                for outputs in outputs_list]
            outputs = dict()

            proposals_all = proposals_list[0].proposal_boxes.tensor
            for i in torch.arange(1, len(outputs_list)):
                proposals_all = torch.cat(
                    (proposals_all, proposals_list[i].proposal_boxes.tensor), 0)
            proposals_list[0].proposal_boxes.tensor = proposals_all
            outputs['proposals'] = proposals_list[0]

            box_delta = torch.cat(box_delta_list, 0)
            box_cls = torch.cat(box_cls_list, 0)
            outputs['box_delta'] = box_delta
            outputs['box_cls'] = box_cls

            if box_reg_var_list[0] is not None:
                box_reg_var = torch.cat(box_reg_var_list, 0)
            else:
                box_reg_var = None
            outputs['box_reg_var'] = box_reg_var

            if box_cls_var_list[0] is not None:
                box_cls_var = torch.cat(box_cls_var_list, 0)
            else:
                box_cls_var = None
            outputs['box_cls_var'] = box_cls_var

        elif outputs is None:
            # outputs = self.model(input_im)

            ####
            raw_output = dict()
            images = self.model.preprocess_image(input_im)
            features = self.model.backbone(images.tensor)

            if self.model.proposal_generator is not None:
                proposals, _ = self.model.proposal_generator(images, features, None)
            # Create raw output dictionary
            raw_output.update({'proposals': proposals[0]})
            # results, _ = self.model.roi_heads(images, features, proposals, None)


            features = [features[f] for f in self.model.roi_heads.box_in_features]
            box_features = self.model.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])
            box_features = self.model.roi_heads.box_head(box_features)
            predictions = self.model.roi_heads.box_predictor(box_features)



            box_cls = predictions[0]
            box_delta = predictions[1]
            box_cls_var = None
            box_reg_var = None
            raw_output.update({'box_cls': box_cls,
                               'box_delta': box_delta,
                               'box_cls_var': box_cls_var,
                               'box_reg_var': box_reg_var})
            outputs = raw_output
            ####


        proposals = outputs['proposals']
        box_cls = outputs['box_cls']
        box_delta = outputs['box_delta']

        inter_feat = box_cls
        box_cls = torch.nn.functional.softmax(box_cls, dim=-1)


        # Remove background category
        scores = box_cls[:, :-1]

        num_bbox_reg_classes = box_delta.shape[1] // 4
        box_delta = box_delta.reshape(-1, 4)
        box_delta = box_delta.view(-1, num_bbox_reg_classes, 4)
        filter_mask = scores > self.test_score_thres

        filter_inds = filter_mask.nonzero(as_tuple=False)

        if num_bbox_reg_classes == 1:
            box_delta = box_delta[filter_inds[:, 0], 0]
        else:
            box_delta = box_delta[filter_mask]

        det_labels = torch.arange(scores.shape[1], dtype=torch.long)
        det_labels = det_labels.view(1, -1).expand_as(scores)

        scores = scores[filter_mask]
        det_labels = det_labels[filter_mask]

        inter_feat = inter_feat[filter_inds[:, 0]]
        proposal_boxes = proposals.proposal_boxes.tensor[filter_inds[:, 0]]

        # predict boxes
        boxes = self.model.roi_heads.box_predictor.box2box_transform.apply_deltas(
            box_delta, proposal_boxes)
        boxes_covars = []


        return boxes, boxes_covars, scores, inter_feat, filter_inds[:,
                                                        1], box_cls[filter_inds[:, 0]], det_labels

    def post_processing_standard_nms(self, input_im):
        """
        This function produces results using standard non-maximum suppression. The function takes into
        account any probabilistic modeling method when computing the results.
        Args:
            input_im (list): an input im list generated from dataset handler.
        Returns:
            result (instances): object instances
        """
        outputs = self.generalized_rcnn_probabilistic_inference(input_im)

        return inference_utils.general_standard_nms_postprocessing(
            input_im, outputs, self.test_nms_thresh, self.test_topk_per_image)

    def post_processing_output_statistics(self, input_im):
        """
        This function produces results using anchor statistics.
        Args:
            input_im (list): an input im list generated from dataset handler.
        Returns:
            result (instances): object instances
        """

        outputs = self.generalized_rcnn_probabilistic_inference(input_im)

        return inference_utils.general_output_statistics_postprocessing(
            input_im,
            outputs,
            self.test_nms_thresh,
            self.test_topk_per_image,
            self.cfg.PROBABILISTIC_INFERENCE.AFFINITY_THRESHOLD)

    def post_processing_mc_dropout_ensembles(self, input_im):
        """
        This function produces results using monte-carlo dropout ensembles.
        Args:
            input_im (list): an input im list generated from dataset handler.
        Returns:
            result (instances): object instances
        """
        if self.cfg.PROBABILISTIC_INFERENCE.ENSEMBLES.BOX_MERGE_MODE == 'pre_nms':
            # In generalized rcnn models, association cannot be achieved on an anchor level when using
            # dropout as anchor order might shift. To overcome this problem, the anchor statistics function
            # is used to perform the association and to fuse covariance
            # results.
            return self.post_processing_output_statistics(input_im)
        else:
            outputs_list = self.model(
                input_im,
                return_anchorwise_output=False,
                num_mc_dropout_runs=self.num_mc_dropout_runs)

            # Merge results:
            results = [
                inference_utils.general_standard_nms_postprocessing(
                    input_im,
                    self.generalized_rcnn_probabilistic_inference(
                        input_im,
                        outputs=outputs),
                    self.test_nms_thresh,
                    self.test_topk_per_image) for outputs in outputs_list]

            # Append per-ensemble outputs after NMS has been performed.
            ensemble_pred_box_list = [
                result.pred_boxes.tensor for result in results]
            ensemble_pred_prob_vectors_list = [
                result.pred_cls_probs for result in results]
            ensembles_class_idxs_list = [
                result.pred_classes for result in results]
            ensembles_pred_box_covariance_list = [
                result.pred_boxes_covariance for result in results]

            return inference_utils.general_black_box_ensembles_post_processing(
                input_im,
                ensemble_pred_box_list,
                ensembles_class_idxs_list,
                ensemble_pred_prob_vectors_list,
                ensembles_pred_box_covariance_list,
                self.test_nms_thresh,
                self.test_topk_per_image,
                self.cfg.PROBABILISTIC_INFERENCE.AFFINITY_THRESHOLD,
                is_generalized_rcnn=True,
                merging_method=self.cfg.PROBABILISTIC_INFERENCE.ENSEMBLES.BOX_FUSION_MODE)

    def post_processing_ensembles(self, input_im, model_dict):
        if self.cfg.PROBABILISTIC_INFERENCE.ENSEMBLES.BOX_MERGE_MODE == 'pre_nms':
            outputs_list = []

            for model in model_dict:
                outputs = model(input_im, return_anchorwise_output=True)
                outputs_list.append(outputs)

            outputs = self.generalized_rcnn_probabilistic_inference(
                input_im, ensemble_inference=True, outputs_list=outputs_list)

            return inference_utils.general_output_statistics_postprocessing(
                input_im,
                outputs,
                self.test_nms_thresh,
                self.test_topk_per_image,
                self.cfg.PROBABILISTIC_INFERENCE.AFFINITY_THRESHOLD)
        else:
            outputs_list = []
            for model in model_dict:
                self.model = model
                outputs_list.append(
                    self.post_processing_standard_nms(input_im))

            # Merge results:
            ensemble_pred_box_list = []
            ensemble_pred_prob_vectors_list = []
            ensembles_class_idxs_list = []
            ensembles_pred_box_covariance_list = []
            for results in outputs_list:
                # Append per-ensemble outputs after NMS has been performed.
                ensemble_pred_box_list.append(results.pred_boxes.tensor)
                ensemble_pred_prob_vectors_list.append(results.pred_cls_probs)
                ensembles_class_idxs_list.append(results.pred_classes)
                ensembles_pred_box_covariance_list.append(
                    results.pred_boxes_covariance)

            return inference_utils.general_black_box_ensembles_post_processing(
                input_im,
                ensemble_pred_box_list,
                ensembles_class_idxs_list,
                ensemble_pred_prob_vectors_list,
                ensembles_pred_box_covariance_list,
                self.test_nms_thresh,
                self.test_topk_per_image,
                self.cfg.PROBABILISTIC_INFERENCE.AFFINITY_THRESHOLD,
                is_generalized_rcnn=True,
                merging_method=self.cfg.PROBABILISTIC_INFERENCE.ENSEMBLES.BOX_FUSION_MODE)

    def post_processing_bayes_od(self, input_im):
        """
        This function produces results using forms of bayesian inference instead of NMS for both category
        and box results.
        Args:
            input_im (list): an input im list generated from dataset handler.
        Returns:
            result (instances): object instances
        """
        box_merge_mode = self.cfg.PROBABILISTIC_INFERENCE.BAYES_OD.BOX_MERGE_MODE
        cls_merge_mode = self.cfg.PROBABILISTIC_INFERENCE.BAYES_OD.CLS_MERGE_MODE

        outputs = self.generalized_rcnn_probabilistic_inference(input_im)

        predicted_boxes, predicted_boxes_covariance, predicted_prob, classes_idxs, predicted_prob_vectors = outputs

        keep = batched_nms(
            predicted_boxes,
            predicted_prob,
            classes_idxs,
            self.test_nms_thresh)

        keep = keep[: self.test_topk_per_image]

        match_quality_matrix = pairwise_iou(
            Boxes(predicted_boxes), Boxes(predicted_boxes))

        box_clusters_inds = match_quality_matrix[keep, :]
        box_clusters_inds = box_clusters_inds > self.cfg.PROBABILISTIC_INFERENCE.AFFINITY_THRESHOLD

        # Compute mean and covariance for every cluster.
        predicted_boxes_list = []
        predicted_boxes_covariance_list = []
        predicted_prob_vectors_list = []

        predicted_prob_vectors_centers = predicted_prob_vectors[keep]
        for box_cluster, predicted_prob_vectors_center in zip(
                box_clusters_inds, predicted_prob_vectors_centers):

            # Ignore background categories provided by detectron2 inference
            cluster_categorical_params = predicted_prob_vectors[box_cluster]
            _, center_cat_idx = torch.max(predicted_prob_vectors_center, 0)
            _, cat_idx = cluster_categorical_params.max(1)
            class_similarity_idx = cat_idx == center_cat_idx

            if cls_merge_mode == 'bayesian_inference':
                cluster_categorical_params = cluster_categorical_params[class_similarity_idx]
                predicted_prob_vectors_list.append(
                    cluster_categorical_params.mean(0).unsqueeze(0))
            else:
                predicted_prob_vectors_list.append(
                    predicted_prob_vectors_center.unsqueeze(0))

            # Switch to numpy as torch.inverse is too slow.
            cluster_means = predicted_boxes[box_cluster,
                                            :][class_similarity_idx].cpu().numpy()
            cluster_covs = predicted_boxes_covariance[box_cluster, :][class_similarity_idx].cpu(
            ).numpy()

            predicted_box, predicted_box_covariance = inference_utils.bounding_box_bayesian_inference(
                cluster_means, cluster_covs, box_merge_mode)
            predicted_boxes_list.append(
                torch.from_numpy(np.squeeze(predicted_box)))
            predicted_boxes_covariance_list.append(
                torch.from_numpy(predicted_box_covariance))

        # Switch back to cuda for the remainder of the inference process.
        result = Instances(
            (input_im[0]['image'].shape[1],
             input_im[0]['image'].shape[2]))

        if len(predicted_boxes_list) > 0:
            if cls_merge_mode == 'bayesian_inference':
                predicted_prob_vectors = torch.cat(
                    predicted_prob_vectors_list, 0)
                predicted_prob, classes_idxs = torch.max(
                    predicted_prob_vectors[:, :-1], 1)
            elif cls_merge_mode == 'max_score':
                predicted_prob_vectors = predicted_prob_vectors[keep]
                predicted_prob = predicted_prob[keep]
                classes_idxs = classes_idxs[keep]

            result.pred_boxes = Boxes(
                torch.stack(
                    predicted_boxes_list,
                    0).to(self.model.device))
            result.scores = predicted_prob
            result.pred_classes = classes_idxs
            result.pred_cls_probs = predicted_prob_vectors
            result.pred_boxes_covariance = torch.stack(
                predicted_boxes_covariance_list, 0).to(self.model.device)
        else:
            result.pred_boxes = Boxes(predicted_boxes)
            result.scores = torch.zeros(
                predicted_boxes.shape[0]).to(
                self.model.device)
            result.pred_classes = classes_idxs
            result.pred_cls_probs = predicted_prob_vectors
            result.pred_boxes_covariance = torch.empty(
                (predicted_boxes.shape + (4,))).to(self.model.device)
        return result