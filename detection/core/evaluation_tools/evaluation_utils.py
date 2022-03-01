import numpy as np
import os
import tqdm
import torch
import ujson as json

from collections import defaultdict

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, pairwise_iou

# Project imports
from core.datasets import metadata

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_predictions_preprocess(
        predicted_instances,
        min_allowed_score=0.0,
        is_odd=False,
        is_gmm=False):
    predicted_boxes, predicted_cls_probs, predicted_covar_mats, predicted_inter_feat, predicted_logistic_score = defaultdict(
        torch.Tensor), defaultdict(
        torch.Tensor), defaultdict(
            torch.Tensor), defaultdict(
            torch.Tensor), defaultdict(torch.Tensor)

    for predicted_instance in predicted_instances:
        # Remove predictions with undefined category_id. This is used when the training and
        # inference datasets come from different data such as COCO-->VOC or COCO-->OpenImages.
        # Only happens if not ODD dataset, else all detections will be removed.

        if len(predicted_instance['cls_prob']) == 81 or len(predicted_instance['cls_prob']) == 21 or len(predicted_instance['cls_prob']) == 11:
            cls_prob = predicted_instance['cls_prob'][:-1]
        else:
            cls_prob = predicted_instance['cls_prob']

        if not is_odd:
            skip_test = (
                predicted_instance['category_id'] == -
                1) or (
                np.array(cls_prob).max(0) < min_allowed_score)
        else:
            skip_test = np.array(cls_prob).max(0) < min_allowed_score

        if skip_test:
            continue

        box_inds = predicted_instance['bbox']
        box_inds = np.array([box_inds[0],
                             box_inds[1],
                             box_inds[0] + box_inds[2],
                             box_inds[1] + box_inds[3]])

        predicted_boxes[predicted_instance['image_id']] = torch.cat((predicted_boxes[predicted_instance['image_id']].to(
            device), torch.as_tensor([box_inds], dtype=torch.float32).to(device)))

        predicted_cls_probs[predicted_instance['image_id']] = torch.cat((predicted_cls_probs[predicted_instance['image_id']].to(
            device), torch.as_tensor([predicted_instance['cls_prob']], dtype=torch.float32).to(device)))

        box_covar = np.array(predicted_instance['bbox_covar'])
        # import ipdb; ipdb.set_trace()
        transformation_mat = np.array([[1.0, 0, 0, 0],
                                       [0, 1.0, 0, 0],
                                       [1.0, 0, 1.0, 0],
                                       [0, 1.0, 0.0, 1.0]])
        cov_pred = np.matmul(
            np.matmul(
                transformation_mat,
                box_covar),
            transformation_mat.T).tolist()

        if not is_gmm:
            if len(predicted_instance['inter_feat']) == 2:
                breakpoint()
                predicted_inter_feat[predicted_instance['image_id']] = torch.cat(
                    (
                        predicted_inter_feat[predicted_instance['image_id']].to(device),
                        torch.as_tensor(predicted_instance['inter_feat'], dtype=torch.float32).to(device)
                    ),
                    0
                )
            else:
                predicted_inter_feat[predicted_instance['image_id']] = torch.cat(
                    (
                        predicted_inter_feat[predicted_instance['image_id']].to(device),
                        torch.as_tensor(predicted_instance['inter_feat'], dtype=torch.float32).to(device).unsqueeze(0)
                    ),
                    0
                )
        else:
            predicted_inter_feat[predicted_instance['image_id']] = torch.cat(
                (
                    predicted_inter_feat[predicted_instance['image_id']].to(device),
                    torch.as_tensor(predicted_instance['inter_feat'], dtype=torch.float32).to(device).unsqueeze(0)
                ),
                0
            )
        # breakpoint()
        # breakpoint()
        if 'logistic_score' in list(predicted_instance.keys()):
            predicted_logistic_score[predicted_instance['image_id']] = torch.cat(
                (
                    predicted_logistic_score[predicted_instance['image_id']].to(device),
                    torch.as_tensor(predicted_instance['logistic_score'], dtype=torch.float32).to(device).unsqueeze(0)
                ),
                0
            )


        predicted_covar_mats[predicted_instance['image_id']] = torch.cat(
            (predicted_covar_mats[predicted_instance['image_id']].to(device), torch.as_tensor([cov_pred], dtype=torch.float32).to(device)))

    if 'logistic_score' not in list(predicted_instance.keys()):
        return dict({'predicted_boxes': predicted_boxes,
                     'predicted_cls_probs': predicted_cls_probs,
                     "predicted_inter_feat": predicted_inter_feat,
                     'predicted_covar_mats': predicted_covar_mats})
    else:
        return dict({'predicted_boxes': predicted_boxes,
                     'predicted_cls_probs': predicted_cls_probs,
                     'predicted_logistic_score': predicted_logistic_score,
                     "predicted_inter_feat": predicted_inter_feat,
                     'predicted_covar_mats': predicted_covar_mats})


def eval_gt_preprocess(gt_instances):
    gt_boxes, gt_cat_idxs, gt_is_truncated, gt_is_occluded = defaultdict(
        torch.Tensor), defaultdict(
        torch.Tensor), defaultdict(
        torch.Tensor), defaultdict(
        torch.Tensor)
    for gt_instance in gt_instances:
        box_inds = gt_instance['bbox']
        box_inds = np.array([box_inds[0],
                             box_inds[1],
                             box_inds[0] + box_inds[2],
                             box_inds[1] + box_inds[3]])
        gt_boxes[gt_instance['image_id']] = torch.cat((gt_boxes[gt_instance['image_id']].cuda(
        ), torch.as_tensor([box_inds], dtype=torch.float32).to(device)))
        gt_cat_idxs[gt_instance['image_id']] = torch.cat((gt_cat_idxs[gt_instance['image_id']].cuda(
        ), torch.as_tensor([[gt_instance['category_id']]], dtype=torch.float32).to(device)))

        if 'is_truncated' in gt_instance.keys():
            gt_is_truncated[gt_instance['image_id']] = torch.cat((gt_is_truncated[gt_instance['image_id']].cuda(
            ), torch.as_tensor([gt_instance['is_truncated']], dtype=torch.float32).to(device)))

            gt_is_occluded[gt_instance['image_id']] = torch.cat((gt_is_occluded[gt_instance['image_id']].cuda(
            ), torch.as_tensor([gt_instance['is_occluded']], dtype=torch.float32).to(device)))

    if 'is_truncated' in gt_instances[0].keys():
        return dict({'gt_boxes': gt_boxes,
                     'gt_cat_idxs': gt_cat_idxs,
                     'gt_is_truncated': gt_is_truncated,
                     'gt_is_occluded': gt_is_occluded})
    else:
        return dict({'gt_boxes': gt_boxes,
                     'gt_cat_idxs': gt_cat_idxs})


def get_matched_results(
        cfg,
        inference_output_dir,
        iou_min=0.1,
        iou_correct=0.7,
        min_allowed_score=0.0):
    try:
        matched_results = torch.load(
            os.path.join(
                inference_output_dir,
                "matched_results_{}_{}_{}.pth".format(
                    iou_min,
                    iou_correct,
                    min_allowed_score)), map_location=device)

        return matched_results
    except FileNotFoundError:
        preprocessed_predicted_instances, preprocessed_gt_instances = get_per_frame_preprocessed_instances(
            cfg, inference_output_dir, min_allowed_score)
        predicted_box_means = preprocessed_predicted_instances['predicted_boxes']
        predicted_cls_probs = preprocessed_predicted_instances['predicted_cls_probs']
        predicted_box_covariances = preprocessed_predicted_instances['predicted_covar_mats']
        gt_box_means = preprocessed_gt_instances['gt_boxes']
        gt_cat_idxs = preprocessed_gt_instances['gt_cat_idxs']

        if 'gt_is_truncated' in preprocessed_gt_instances.keys():
            is_truncated = preprocessed_gt_instances['gt_is_truncated']
        else:
            is_truncated = None

        if 'gt_is_occluded' in preprocessed_gt_instances.keys():
            is_occluded = preprocessed_gt_instances['gt_is_occluded']
        else:
            is_occluded = None

        matched_results = match_predictions_to_groundtruth(
            predicted_box_means,
            predicted_cls_probs,
            predicted_box_covariances,
            gt_box_means,
            gt_cat_idxs,
            iou_min,
            iou_correct,
            is_truncated=is_truncated,
            is_occluded=is_occluded)

        torch.save(
            matched_results,
            os.path.join(
                inference_output_dir,
                "matched_results_{}_{}_{}.pth".format(
                    iou_min,
                    iou_correct,
                    min_allowed_score)))

        return matched_results


def get_per_frame_preprocessed_instances(
        cfg, inference_output_dir, min_allowed_score=0.0):
    prediction_file_name = os.path.join(
        inference_output_dir,
        'coco_instances_results.json')

    meta_catalog = MetadataCatalog.get(cfg.ACTUAL_TEST_DATASET)
    # Process GT
    print("Began pre-processing ground truth annotations...")
    try:
        preprocessed_gt_instances = torch.load(
            os.path.join(
                os.path.split(meta_catalog.json_file)[0],
                "preprocessed_gt_instances.pth"), map_location=device)
    except FileNotFoundError:
        gt_info = json.load(
            open(
                meta_catalog.json_file,
                'r'))
        gt_instances = gt_info['annotations']
        preprocessed_gt_instances = eval_gt_preprocess(
            gt_instances)
        torch.save(
            preprocessed_gt_instances,
            os.path.join(
                os.path.split(meta_catalog.json_file)[0],
                "preprocessed_gt_instances.pth"))
    print("Done!")
    print("Began pre-processing predicted instances...")
    try:
        preprocessed_predicted_instances = torch.load(
            os.path.join(
                inference_output_dir,
                "preprocessed_predicted_instances_{}.pth".format(min_allowed_score)),
            map_location=device)
    # Process predictions
    except FileNotFoundError:
        predicted_instances = json.load(open(prediction_file_name, 'r'))
        preprocessed_predicted_instances = eval_predictions_preprocess(
            predicted_instances, min_allowed_score)
        torch.save(
            preprocessed_predicted_instances,
            os.path.join(
                inference_output_dir,
                "preprocessed_predicted_instances_{}.pth".format(min_allowed_score)))
    print("Done!")

    return preprocessed_predicted_instances, preprocessed_gt_instances


def match_predictions_to_groundtruth(predicted_box_means,
                                     predicted_cls_probs,
                                     predicted_box_covariances,
                                     gt_box_means,
                                     gt_cat_idxs,
                                     iou_min=0.1,
                                     iou_correct=0.7,
                                     is_truncated=None,
                                     is_occluded=None):

    # Flag to know if truncation and occlusion should be saved:
    trunc_occ_flag = is_truncated is not None and is_occluded is not None

    true_positives = dict(
        {
            'predicted_box_means': torch.Tensor().to(device),
            'predicted_box_covariances': torch.Tensor().to(device),
            'predicted_cls_probs': torch.Tensor().to(device),
            'gt_box_means': torch.Tensor().to(device),
            'gt_cat_idxs': torch.Tensor().to(device),
            'iou_with_ground_truth': torch.Tensor().to(device),
            'is_truncated': torch.Tensor().to(device),
            'is_occluded': torch.Tensor().to(device)})

    localization_errors = dict(
        {
            'predicted_box_means': torch.Tensor().to(device),
            'predicted_box_covariances': torch.Tensor().to(device),
            'predicted_cls_probs': torch.Tensor().to(device),
            'gt_box_means': torch.Tensor().to(device),
            'gt_cat_idxs': torch.Tensor().to(device),
            'iou_with_ground_truth': torch.Tensor().to(device),
            'is_truncated': torch.Tensor().to(device),
            'is_occluded': torch.Tensor().to(device)})

    duplicates = dict({'predicted_box_means': torch.Tensor().to(device),
                       'predicted_box_covariances': torch.Tensor().to(device),
                       'predicted_cls_probs': torch.Tensor().to(device),
                       'gt_box_means': torch.Tensor().to(device),
                       'gt_cat_idxs': torch.Tensor().to(device),
                       'iou_with_ground_truth': torch.Tensor().to(device),
                       'is_truncated': torch.Tensor().to(device),
                       'is_occluded': torch.Tensor().to(device)})

    false_positives = dict({'predicted_box_means': torch.Tensor().to(device),
                            'predicted_box_covariances': torch.Tensor().to(device),
                            'predicted_cls_probs': torch.Tensor().to(device)})

    false_negatives = dict({'gt_box_means': torch.Tensor().to(device),
                            'gt_cat_idxs': torch.Tensor().to(device),
                            'is_truncated': torch.Tensor().to(device),
                            'is_occluded': torch.Tensor().to(device)
                            })

    with tqdm.tqdm(total=len(predicted_box_means)) as pbar:
        for key in predicted_box_means.keys():
            pbar.update(1)

            # Check if gt available, if not all detections go to false
            # positives
            if key not in gt_box_means.keys():
                false_positives['predicted_box_means'] = torch.cat(
                    (false_positives['predicted_box_means'], predicted_box_means[key]))
                false_positives['predicted_cls_probs'] = torch.cat(
                    (false_positives['predicted_cls_probs'], predicted_cls_probs[key]))
                false_positives['predicted_box_covariances'] = torch.cat(
                    (false_positives['predicted_box_covariances'], predicted_box_covariances[key]))
                continue

            # Compute iou between gt boxes and all predicted boxes in frame
            frame_gt_boxes = Boxes(gt_box_means[key])
            frame_predicted_boxes = Boxes(predicted_box_means[key])
            num_predictions_in_frame = frame_predicted_boxes.tensor.shape[0]

            match_iou = pairwise_iou(frame_gt_boxes, frame_predicted_boxes)

            # False positives are detections that have an iou < match iou with
            # any ground truth object.
            false_positive_idxs = (match_iou <= iou_min).all(0)
            false_positives['predicted_box_means'] = torch.cat(
                (false_positives['predicted_box_means'],
                 predicted_box_means[key][false_positive_idxs]))
            false_positives['predicted_cls_probs'] = torch.cat(
                (false_positives['predicted_cls_probs'],
                 predicted_cls_probs[key][false_positive_idxs]))
            false_positives['predicted_box_covariances'] = torch.cat(
                (false_positives['predicted_box_covariances'],
                 predicted_box_covariances[key][false_positive_idxs]))

            num_fp_in_frame = false_positive_idxs.sum(0)

            # True positives are any detections with match iou > iou correct. We need to separate these detections to
            # True positive and duplicate set. The true positive detection is the detection assigned the highest score
            # by the neural network.
            true_positive_idxs = torch.nonzero(
                match_iou >= iou_correct, as_tuple=False)

            # Setup tensors to allow assignment of detections only once.
            processed_gt = torch.tensor([]).type(torch.LongTensor).to(device)
            predictions_idxs_processed = torch.tensor(
                []).type(torch.LongTensor).to(device)

            for i in torch.arange(frame_gt_boxes.tensor.shape[0]):
                # Check if true positive has been previously assigned to a ground truth box and remove it if this is
                # the case. Very rare occurrence but need to handle it
                # nevertheless.
                prediction_idxs = true_positive_idxs[true_positive_idxs[:, 0] == i][:, 1]
                non_valid_idxs = torch.nonzero(
                    predictions_idxs_processed[..., None] == prediction_idxs, as_tuple=False)

                if non_valid_idxs.shape[0] > 0:
                    prediction_idxs[non_valid_idxs[:, 1]] = -1
                    prediction_idxs = prediction_idxs[prediction_idxs != -1]

                if prediction_idxs.shape[0] > 0:
                    # If there is a prediction attached to gt, count it as
                    # processed.
                    processed_gt = torch.cat(
                        (processed_gt, i.unsqueeze(0).to(
                            processed_gt.device)))
                    predictions_idxs_processed = torch.cat(
                        (predictions_idxs_processed, prediction_idxs))

                    current_matches_predicted_cls_probs = predicted_cls_probs[key][prediction_idxs]
                    max_score, _ = torch.max(
                        current_matches_predicted_cls_probs, 1)
                    _, max_idxs = max_score.topk(max_score.shape[0])

                    if max_idxs.shape[0] > 1:
                        max_idx = max_idxs[0]
                        duplicate_idxs = max_idxs[1:]
                    else:
                        max_idx = max_idxs
                        duplicate_idxs = torch.empty(0).to(device)

                    current_matches_predicted_box_means = predicted_box_means[key][prediction_idxs]
                    current_matches_predicted_box_covariances = predicted_box_covariances[
                        key][prediction_idxs]

                    # Highest scoring detection goes to true positives
                    true_positives['predicted_box_means'] = torch.cat(
                        (true_positives['predicted_box_means'],
                         current_matches_predicted_box_means[max_idx:max_idx + 1, :]))
                    true_positives['predicted_cls_probs'] = torch.cat(
                        (true_positives['predicted_cls_probs'],
                         current_matches_predicted_cls_probs[max_idx:max_idx + 1, :]))
                    true_positives['predicted_box_covariances'] = torch.cat(
                        (true_positives['predicted_box_covariances'],
                         current_matches_predicted_box_covariances[max_idx:max_idx + 1, :]))

                    true_positives['gt_box_means'] = torch.cat(
                        (true_positives['gt_box_means'], gt_box_means[key][i:i + 1, :]))
                    true_positives['gt_cat_idxs'] = torch.cat(
                        (true_positives['gt_cat_idxs'], gt_cat_idxs[key][i:i + 1, :]))
                    if trunc_occ_flag:
                        true_positives['is_truncated'] = torch.cat(
                            (true_positives['is_truncated'], is_truncated[key][i:i + 1]))
                        true_positives['is_occluded'] = torch.cat(
                            (true_positives['is_occluded'], is_occluded[key][i:i + 1]))
                    true_positives['iou_with_ground_truth'] = torch.cat(
                        (true_positives['iou_with_ground_truth'], match_iou[i, prediction_idxs][max_idx:max_idx + 1]))

                    # Lower scoring redundant detections go to duplicates
                    if duplicate_idxs.shape[0] > 1:
                        duplicates['predicted_box_means'] = torch.cat(
                            (duplicates['predicted_box_means'], current_matches_predicted_box_means[duplicate_idxs, :]))
                        duplicates['predicted_cls_probs'] = torch.cat(
                            (duplicates['predicted_cls_probs'], current_matches_predicted_cls_probs[duplicate_idxs, :]))
                        duplicates['predicted_box_covariances'] = torch.cat(
                            (duplicates['predicted_box_covariances'],
                             current_matches_predicted_box_covariances[duplicate_idxs, :]))

                        duplicates['gt_box_means'] = torch.cat(
                            (duplicates['gt_box_means'], gt_box_means[key][np.repeat(i, duplicate_idxs.shape[0]), :]))
                        duplicates['gt_cat_idxs'] = torch.cat(
                            (duplicates['gt_cat_idxs'], gt_cat_idxs[key][np.repeat(i, duplicate_idxs.shape[0]), :]))
                        if trunc_occ_flag:
                            duplicates['is_truncated'] = torch.cat(
                                (duplicates['is_truncated'], is_truncated[key][np.repeat(i, duplicate_idxs.shape[0])]))
                            duplicates['is_occluded'] = torch.cat(
                                (duplicates['is_occluded'], is_occluded[key][np.repeat(i, duplicate_idxs.shape[0])]))
                        duplicates['iou_with_ground_truth'] = torch.cat(
                            (duplicates['iou_with_ground_truth'],
                             match_iou[i, prediction_idxs][duplicate_idxs]))

                    elif duplicate_idxs.shape[0] == 1:
                        # Special case when only one duplicate exists, required to
                        # index properly for torch.cat
                        duplicates['predicted_box_means'] = torch.cat(
                            (duplicates['predicted_box_means'],
                             current_matches_predicted_box_means[duplicate_idxs:duplicate_idxs + 1, :]))
                        duplicates['predicted_cls_probs'] = torch.cat(
                            (duplicates['predicted_cls_probs'],
                             current_matches_predicted_cls_probs[duplicate_idxs:duplicate_idxs + 1, :]))
                        duplicates['predicted_box_covariances'] = torch.cat(
                            (duplicates['predicted_box_covariances'],
                             current_matches_predicted_box_covariances[duplicate_idxs:duplicate_idxs + 1, :]))

                        duplicates['gt_box_means'] = torch.cat(
                            (duplicates['gt_box_means'], gt_box_means[key][i:i + 1, :]))
                        duplicates['gt_cat_idxs'] = torch.cat(
                            (duplicates['gt_cat_idxs'], gt_cat_idxs[key][i:i + 1, :]))
                        if trunc_occ_flag:
                            duplicates['is_truncated'] = torch.cat(
                                (duplicates['is_truncated'], is_truncated[key][i:i + 1]))
                            duplicates['is_occluded'] = torch.cat(
                                (duplicates['is_occluded'], is_occluded[key][i:i + 1]))
                        duplicates['iou_with_ground_truth'] = torch.cat(
                            (duplicates['iou_with_ground_truth'],
                             match_iou[i, prediction_idxs][duplicate_idxs:duplicate_idxs + 1]))
            num_tp_dup_in_frame = predictions_idxs_processed.shape[0]
            # Process localization errors. Localization errors are detections with iou < 0.5 with any ground truth.
            # Mask out processed true positives/duplicates so they are not
            # re-associated with another gt
            # ToDo Localization Errors and False Positives are constant, do not change. We could generate them only
            # once.
            match_iou[:, true_positive_idxs[:, 1]] *= 0.0

            localization_errors_idxs = torch.nonzero(
                (match_iou > iou_min) & (
                    match_iou < 0.5), as_tuple=False)

            # Setup tensors to allow assignment of detections only once.
            processed_localization_errors = torch.tensor(
                []).type(torch.LongTensor).to(device)

            for localization_error_idx in localization_errors_idxs[:, 1]:
                # If localization error has been processed, skip iteration.
                if (processed_localization_errors ==
                        localization_error_idx).any():
                    continue
                # For every localization error, assign the ground truth with
                # highest IOU.
                gt_loc_error_idxs = localization_errors_idxs[localization_errors_idxs[:, 1]
                                                             == localization_error_idx]
                ious_with_gts = match_iou[gt_loc_error_idxs[:,
                                                            0], gt_loc_error_idxs[:, 1]]
                gt_loc_error_idxs = gt_loc_error_idxs[:, 0]

                # Choose the gt with the largest IOU with localization error
                if gt_loc_error_idxs.shape[0] > 1:
                    sorted_idxs = ious_with_gts.sort(
                        descending=True)[1]
                    gt_loc_error_idxs = gt_loc_error_idxs[sorted_idxs[0]:sorted_idxs[0] + 1]

                processed_gt = torch.cat((processed_gt,
                                          gt_loc_error_idxs))

                localization_errors['predicted_box_means'] = torch.cat(
                    (localization_errors['predicted_box_means'],
                     predicted_box_means[key][localization_error_idx:localization_error_idx + 1, :]))
                localization_errors['predicted_cls_probs'] = torch.cat(
                    (localization_errors['predicted_cls_probs'],
                     predicted_cls_probs[key][localization_error_idx:localization_error_idx + 1, :]))
                localization_errors['predicted_box_covariances'] = torch.cat(
                    (localization_errors['predicted_box_covariances'],
                     predicted_box_covariances[key][localization_error_idx:localization_error_idx + 1, :]))

                localization_errors['gt_box_means'] = torch.cat(
                    (localization_errors['gt_box_means'], gt_box_means[key][gt_loc_error_idxs:gt_loc_error_idxs + 1, :]))
                localization_errors['gt_cat_idxs'] = torch.cat(
                    (localization_errors['gt_cat_idxs'], gt_cat_idxs[key][gt_loc_error_idxs:gt_loc_error_idxs + 1]))
                if trunc_occ_flag:
                    localization_errors['is_truncated'] = torch.cat(
                        (localization_errors['is_truncated'], is_truncated[key][gt_loc_error_idxs:gt_loc_error_idxs + 1]))
                    localization_errors['is_occluded'] = torch.cat(
                        (localization_errors['is_occluded'], is_occluded[key][gt_loc_error_idxs:gt_loc_error_idxs + 1]))

                localization_errors['iou_with_ground_truth'] = torch.cat(
                    (localization_errors['iou_with_ground_truth'],
                     match_iou[gt_loc_error_idxs, localization_error_idx:localization_error_idx + 1]))

                # Append processed localization errors
                processed_localization_errors = torch.cat(
                    (processed_localization_errors, localization_error_idx.unsqueeze(0)))

            # Assert that the total number of processed predictions do not exceed the number of predictions in frame.
            num_loc_errors_in_frame = processed_localization_errors.shape[0]
            num_processed_predictions = num_loc_errors_in_frame + \
                num_fp_in_frame + num_tp_dup_in_frame

            # At the limit where iou_correct=0.5, equality holds.
            assert (num_processed_predictions <= num_predictions_in_frame)

            # Get false negative ground truth, which are fully missed.
            # These can be found by looking for GT instances not processed.
            processed_gt = processed_gt.unique()
            false_negative_idxs = torch.ones(frame_gt_boxes.tensor.shape[0])
            false_negative_idxs[processed_gt] = 0
            false_negative_idxs = false_negative_idxs.type(torch.bool)
            false_negatives['gt_box_means'] = torch.cat(
                (false_negatives['gt_box_means'],
                 gt_box_means[key][false_negative_idxs]))
            false_negatives['gt_cat_idxs'] = torch.cat(
                (false_negatives['gt_cat_idxs'],
                 gt_cat_idxs[key][false_negative_idxs]))
            if trunc_occ_flag:
                false_negatives['is_truncated'] = torch.cat(
                    (false_negatives['is_truncated'],
                     is_truncated[key][false_negative_idxs]))
                false_negatives['is_occluded'] = torch.cat(
                    (false_negatives['is_occluded'],
                     is_occluded[key][false_negative_idxs]))

    matched_results = dict()
    matched_results.update({"true_positives": true_positives,
                            "localization_errors": localization_errors,
                            "duplicates": duplicates,
                            "false_positives": false_positives,
                            "false_negatives": false_negatives})
    return matched_results


def get_train_contiguous_id_to_test_thing_dataset_id_dict(
        cfg,
        args,
        train_thing_dataset_id_to_contiguous_id,
        test_thing_dataset_id_to_contiguous_id):

    # If both dicts are equal or if we are performing out of distribution
    # detection, just flip the test dict.
    if train_thing_dataset_id_to_contiguous_id == test_thing_dataset_id_to_contiguous_id:
        # import ipdb;
        # ipdb.set_trace()
        cat_mapping_dict = dict(
            (v, k) for k, v in test_thing_dataset_id_to_contiguous_id.items())
    else:
        # import ipdb;
        # ipdb.set_trace()
        # If not equal, three situations: 1) BDD to KITTI, 2) COCO to PASCAL,
        # or 3) COCO to OpenImages
        if 'coco_ood_val' == args.test_dataset and 'voc_custom_train' == cfg.DATASETS.TRAIN[0]:
            from collections import ChainMap
            cat_mapping_dict = dict(
                        ChainMap(*[{i: i + 1} for i in range(20)]))
        elif 'voc_ood_val' == args.test_dataset and 'voc_custom_train_id' == cfg.DATASETS.TRAIN[0]:
            import ipdb; ipdb.set_trace()
            from collections import ChainMap
            cat_mapping_dict = dict(
                        ChainMap(*[{i: i + 1} for i in range(10)]))
        elif 'openimages_ood_val' == args.test_dataset and 'voc_custom_train' == cfg.DATASETS.TRAIN[0]:
            from collections import ChainMap
            cat_mapping_dict = dict(
                ChainMap(*[{i: i + 1} for i in range(20)]))
        elif 'coco_ood_val_bdd' == args.test_dataset and 'bdd_custom_train' == cfg.DATASETS.TRAIN[0]:
            from collections import ChainMap
            cat_mapping_dict = dict(
                ChainMap(*[{i: i + 1} for i in range(10)]))
        elif 'openimages_ood_val' == args.test_dataset and 'bdd_custom_train' == cfg.DATASETS.TRAIN[0]:
            from collections import ChainMap
            cat_mapping_dict = dict(
                ChainMap(*[{i: i + 1} for i in range(10)]))
        elif 'voc_custom_val_ood' == args.test_dataset and 'voc_custom_train' == cfg.DATASETS.TRAIN[0]:
            from collections import ChainMap
            cat_mapping_dict = dict(
                ChainMap(*[{i: i + 1} for i in range(20)]))
        else:
            cat_mapping_dict = dict(
                (v, k) for k, v in test_thing_dataset_id_to_contiguous_id.items())
            if 'voc' in args.test_dataset and 'coco' in cfg.DATASETS.TRAIN[0]:
                dataset_mapping_dict = dict(
                    (v, k) for k, v in metadata.COCO_TO_VOC_CONTIGUOUS_ID.items())
            if 'openimages' in args.test_dataset and 'coco' in cfg.DATASETS.TRAIN[0]:
                dataset_mapping_dict = dict(
                    (v, k) for k, v in metadata.COCO_TO_OPENIMAGES_CONTIGUOUS_ID.items())
            elif 'kitti' in args.test_dataset and 'bdd' in cfg.DATASETS.TRAIN[0]:
                dataset_mapping_dict = dict(
                    (v, k) for k, v in metadata.BDD_TO_KITTI_CONTIGUOUS_ID.items())
            else:
                ValueError(
                    'Cannot generate category mapping dictionary. Please check if training and inference datasets are compatible.')
            cat_mapping_dict = dict(
                (dataset_mapping_dict[k], v) for k, v in cat_mapping_dict.items())
        # import ipdb;
        # ipdb.set_trace()
    # breakpoint()
    return cat_mapping_dict


def get_test_thing_dataset_id_to_train_contiguous_id_dict(
        cfg,
        args,
        train_thing_dataset_id_to_contiguous_id,
        test_thing_dataset_id_to_contiguous_id):

    cat_mapping_dict = get_train_contiguous_id_to_test_thing_dataset_id_dict(
        cfg,
        args,
        train_thing_dataset_id_to_contiguous_id,
        test_thing_dataset_id_to_contiguous_id)

    return {v: k for k, v in cat_mapping_dict.items()}
