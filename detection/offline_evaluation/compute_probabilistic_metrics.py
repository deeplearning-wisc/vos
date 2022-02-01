import numpy as np
import os
import torch
import pickle

from prettytable import PrettyTable

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.engine import launch

# Project imports
from core.evaluation_tools import evaluation_utils
from core.evaluation_tools import scoring_rules
from core.evaluation_tools.evaluation_utils import get_test_thing_dataset_id_to_train_contiguous_id_dict
from core.setup import setup_config, setup_arg_parser
from inference.inference_utils import get_inference_output_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
        args,
        cfg=None,
        iou_min=None,
        iou_correct=None,
        min_allowed_score=None,
        print_results=True):

    # Setup config
    if cfg is None:
        cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)

    cfg.defrost()
    cfg.ACTUAL_TEST_DATASET = args.test_dataset

    # Setup torch device and num_threads
    torch.set_num_threads(cfg.DATALOADER.NUM_WORKERS)

    # Build path to gt instances and inference output
    inference_output_dir = get_inference_output_dir(
        cfg['OUTPUT_DIR'],
        args.test_dataset,
        args.inference_config,
        args.image_corruption_level)

    # Get thresholds to perform evaluation on
    if iou_min is None:
        iou_min = args.iou_min
    if iou_correct is None:
        iou_correct = args.iou_correct

    if min_allowed_score is None:
        # Check if F-1 Score has been previously computed ON THE ORIGINAL
        # DATASET such as COCO even when evaluating on OpenImages.
        try:
            train_set_inference_output_dir = get_inference_output_dir(
                cfg['OUTPUT_DIR'],
                cfg.DATASETS.TEST[0],
                args.inference_config,
                0)
            with open(os.path.join(train_set_inference_output_dir, "mAP_res.txt"), "r") as f:
                min_allowed_score = f.read().strip('][\n').split(', ')[-1]
                min_allowed_score = round(float(min_allowed_score), 4)
        except FileNotFoundError:
            # If not, process all detections. Not recommended as the results might be influenced by very low scoring
            # detections that would normally be removed in robotics/vision
            # applications.
            min_allowed_score = 0.0
    # Get category mapping dictionary:
    train_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
        cfg.DATASETS.TRAIN[0]).thing_dataset_id_to_contiguous_id
    test_thing_dataset_id_to_contiguous_id = MetadataCatalog.get(
        args.test_dataset).thing_dataset_id_to_contiguous_id

    cat_mapping_dict = get_test_thing_dataset_id_to_train_contiguous_id_dict(
        cfg,
        args,
        train_thing_dataset_id_to_contiguous_id,
        test_thing_dataset_id_to_contiguous_id)

    # Get matched results by either generating them or loading from file.
    with torch.no_grad():
        matched_results = evaluation_utils.get_matched_results(
            cfg, inference_output_dir,
            iou_min=iou_min,
            iou_correct=iou_correct,
            min_allowed_score=min_allowed_score)

        # Build preliminary dicts required for computing classification scores.
        for matched_results_key in matched_results.keys():
            if 'gt_cat_idxs' in matched_results[matched_results_key].keys():
                # First we convert the written things indices to contiguous
                # indices.
                gt_converted_cat_idxs = matched_results[matched_results_key]['gt_cat_idxs'].squeeze(
                    1)
                gt_converted_cat_idxs = torch.as_tensor([cat_mapping_dict[class_idx.cpu(
                ).tolist()] for class_idx in gt_converted_cat_idxs]).to(device)
                matched_results[matched_results_key]['gt_converted_cat_idxs'] = gt_converted_cat_idxs.to(
                    device)
                if 'predicted_cls_probs' in matched_results[matched_results_key].keys(
                ):
                    predicted_cls_probs = matched_results[matched_results_key]['predicted_cls_probs']
                    # This is required for evaluation of retinanet based
                    # detections.
                    matched_results[matched_results_key]['predicted_score_of_gt_category'] = torch.gather(
                        predicted_cls_probs, 1, gt_converted_cat_idxs.unsqueeze(1)).squeeze(1)
                matched_results[matched_results_key]['gt_cat_idxs'] = gt_converted_cat_idxs
            else:
                if cfg.MODEL.META_ARCHITECTURE == 'ProbabilisticRetinaNet':
                    # For false positives, the correct category is background. For retinanet, since no explicit
                    # background category is available, this value is computed as 1.0 - score of the predicted
                    # category.
                    predicted_class_probs, predicted_class_idx = matched_results[matched_results_key]['predicted_cls_probs'].max(
                        1)
                    matched_results[matched_results_key]['predicted_score_of_gt_category'] = 1.0 - \
                        predicted_class_probs
                    matched_results[matched_results_key]['predicted_cat_idxs'] = predicted_class_idx
                else:
                    # For RCNN/DETR based networks, a background category is
                    # explicitly available.
                    matched_results[matched_results_key]['predicted_score_of_gt_category'] = matched_results[
                        matched_results_key]['predicted_cls_probs'][:, -1]
                    _, predicted_class_idx = matched_results[matched_results_key]['predicted_cls_probs'][:, :-1].max(
                        1)
                    matched_results[matched_results_key]['predicted_cat_idxs'] = predicted_class_idx

        # Load the different detection partitions
        true_positives = matched_results['true_positives']
        duplicates = matched_results['duplicates']
        localization_errors = matched_results['localization_errors']
        false_negatives = matched_results['false_negatives']
        false_positives = matched_results['false_positives']

        # Get the number of elements in each partition
        num_true_positives = true_positives['predicted_box_means'].shape[0]
        num_duplicates = duplicates['predicted_box_means'].shape[0]
        num_localization_errors = localization_errors['predicted_box_means'].shape[0]
        num_false_negatives = false_negatives['gt_box_means'].shape[0]
        num_false_positives = false_positives['predicted_box_means'].shape[0]

        per_class_output_list = []
        for class_idx in cat_mapping_dict.values():
            true_positives_valid_idxs = true_positives['gt_converted_cat_idxs'] == class_idx
            localization_errors_valid_idxs = localization_errors['gt_converted_cat_idxs'] == class_idx
            duplicates_valid_idxs = duplicates['gt_converted_cat_idxs'] == class_idx
            false_positives_valid_idxs = false_positives['predicted_cat_idxs'] == class_idx

            if cfg.MODEL.META_ARCHITECTURE == 'ProbabilisticRetinaNet':
                # Compute classification metrics for every partition
                true_positives_cls_analysis = scoring_rules.sigmoid_compute_cls_scores(
                    true_positives, true_positives_valid_idxs)
                localization_errors_cls_analysis = scoring_rules.sigmoid_compute_cls_scores(
                    localization_errors, localization_errors_valid_idxs)
                duplicates_cls_analysis = scoring_rules.sigmoid_compute_cls_scores(
                    duplicates, duplicates_valid_idxs)
                false_positives_cls_analysis = scoring_rules.sigmoid_compute_cls_scores(
                    false_positives, false_positives_valid_idxs)

            else:
                # Compute classification metrics for every partition
                true_positives_cls_analysis = scoring_rules.softmax_compute_cls_scores(
                    true_positives, true_positives_valid_idxs)
                localization_errors_cls_analysis = scoring_rules.softmax_compute_cls_scores(
                    localization_errors, localization_errors_valid_idxs)
                duplicates_cls_analysis = scoring_rules.softmax_compute_cls_scores(
                    duplicates, duplicates_valid_idxs)
                false_positives_cls_analysis = scoring_rules.softmax_compute_cls_scores(
                    false_positives, false_positives_valid_idxs)

            # Compute regression metrics for every partition
            true_positives_reg_analysis = scoring_rules.compute_reg_scores(
                true_positives, true_positives_valid_idxs)
            localization_errors_reg_analysis = scoring_rules.compute_reg_scores(
                localization_errors, localization_errors_valid_idxs)
            duplicates_reg_analysis = scoring_rules.compute_reg_scores(
                duplicates, duplicates_valid_idxs)
            false_positives_reg_analysis = scoring_rules.compute_reg_scores_fn(
                false_positives, false_positives_valid_idxs, False)

            per_class_output_list.append(
                {'true_positives_cls_analysis': true_positives_cls_analysis,
                 'true_positives_reg_analysis': true_positives_reg_analysis,
                 'localization_errors_cls_analysis': localization_errors_cls_analysis,
                 'localization_errors_reg_analysis': localization_errors_reg_analysis,
                 'duplicates_cls_analysis': duplicates_cls_analysis,
                 'duplicates_reg_analysis': duplicates_reg_analysis,
                 'false_positives_cls_analysis': false_positives_cls_analysis,
                 'false_positives_reg_analysis': false_positives_reg_analysis})

        final_accumulated_output_dict = dict()
        final_average_output_dict = dict()

        for key in per_class_output_list[0].keys():
            average_output_dict = dict()
            for inner_key in per_class_output_list[0][key].keys():
                collected_values = [per_class_output[key][inner_key] if per_class_output[key][
                    inner_key] is not None else np.NaN for per_class_output in per_class_output_list]
                collected_values = np.array(collected_values)

                if key in average_output_dict.keys():
                    # Use nan mean since some classes do not have duplicates for
                    # instance or has one duplicate for instance. torch.std returns nan in that case
                    # so we handle those here. This should not have any effect on the final results, as
                    # it only affects inter-class variance which we do not
                    # report anyways.
                    # try:
                    if 'total_entropy' != inner_key:
                        average_output_dict[key].update(
                            {inner_key: np.nanmean(collected_values),
                             inner_key + '_std': np.nanstd(collected_values, ddof=1)})
                    # except:
                    #     import ipdb; ipdb.set_trace()
                    final_accumulated_output_dict[key].update(
                        {inner_key: collected_values})
                else:
                    average_output_dict.update(
                        {key: {inner_key: np.nanmean(collected_values),
                               inner_key + '_std': np.nanstd(collected_values, ddof=1)}})
                    final_accumulated_output_dict.update(
                        {key: {inner_key: collected_values}})
            final_average_output_dict.update(average_output_dict)

        final_accumulated_output_dict.update(
            {
                "num_instances": {
                    "num_true_positives": num_true_positives,
                    "num_duplicates": num_duplicates,
                    "num_localization_errors": num_localization_errors,
                    "num_false_positives": num_false_positives,
                    "num_false_negatives": num_false_negatives}})

        if print_results:
            # Summarize and print all
            table = PrettyTable()
            table.field_names = (['Output Type',
                                  'Number of Instances',
                                  'Cls Negative Log Likelihood',
                                  'Cls Brier Score',
                                  'Reg TP Negative Log Likelihood / FP Entropy',
                                  'Reg Energy Score'])
            table.add_row(
                [
                    "True Positives:",
                    num_true_positives,
                    '{:.4f} ± {:.4f}'.format(
                        final_average_output_dict['true_positives_cls_analysis']['ignorance_score_mean'],
                        final_average_output_dict['true_positives_cls_analysis']['ignorance_score_mean_std']),
                    '{:.4f} ± {:.4f}'.format(
                        final_average_output_dict['true_positives_cls_analysis']['brier_score_mean'],
                        final_average_output_dict['true_positives_cls_analysis']['brier_score_mean_std']),
                    '{:.4f} ± {:.4f}'.format(
                        final_average_output_dict['true_positives_reg_analysis']['ignorance_score_mean'],
                        final_average_output_dict['true_positives_reg_analysis']['ignorance_score_mean_std']),
                    '{:.4f} ± {:.4f}'.format(
                        final_average_output_dict['true_positives_reg_analysis']['energy_score_mean'],
                        final_average_output_dict['true_positives_reg_analysis']['energy_score_mean_std'])])
            table.add_row(
                [
                    "Duplicates:",
                    num_duplicates,
                    '{:.4f} ± {:.4f}'.format(
                        final_average_output_dict['duplicates_cls_analysis']['ignorance_score_mean'],
                        final_average_output_dict['duplicates_cls_analysis']['ignorance_score_mean_std']),
                    '{:.4f} ± {:.4f}'.format(
                        final_average_output_dict['duplicates_cls_analysis']['brier_score_mean'],
                        final_average_output_dict['duplicates_cls_analysis']['brier_score_mean_std']),
                    '{:.4f} ± {:.4f}'.format(
                        final_average_output_dict['duplicates_reg_analysis']['ignorance_score_mean'],
                        final_average_output_dict['duplicates_reg_analysis']['ignorance_score_mean_std']),
                    '{:.4f} ± {:.4f}'.format(
                        final_average_output_dict['duplicates_reg_analysis']['energy_score_mean'],
                        final_average_output_dict['duplicates_reg_analysis']['energy_score_mean_std'])])
            table.add_row(
                [
                    "Localization Errors:",
                    num_localization_errors,
                    '{:.4f} ± {:.4f}'.format(
                        final_average_output_dict['localization_errors_cls_analysis']['ignorance_score_mean'],
                        final_average_output_dict['localization_errors_cls_analysis']['ignorance_score_mean_std']),
                    '{:.4f} ± {:.4f}'.format(
                        final_average_output_dict['localization_errors_cls_analysis']['brier_score_mean'],
                        final_average_output_dict['localization_errors_cls_analysis']['brier_score_mean_std']),
                    '{:.4f} ± {:.4f}'.format(
                        final_average_output_dict['localization_errors_reg_analysis']['ignorance_score_mean'],
                        final_average_output_dict['localization_errors_reg_analysis']['ignorance_score_mean_std']),
                    '{:.4f} ± {:.4f}'.format(
                        final_average_output_dict['localization_errors_reg_analysis']['energy_score_mean'],
                        final_average_output_dict['localization_errors_reg_analysis']['energy_score_mean_std'])])
            table.add_row(
                [
                    "False Positives:",
                    num_false_positives,
                    '{:.4f} ± {:.4f}'.format(
                        final_average_output_dict['false_positives_cls_analysis']['ignorance_score_mean'],
                        final_average_output_dict['false_positives_cls_analysis']['ignorance_score_mean_std']),
                    '{:.4f} ± {:.4f}'.format(
                        final_average_output_dict['false_positives_cls_analysis']['brier_score_mean'],
                        final_average_output_dict['false_positives_cls_analysis']['brier_score_mean_std']),
                    '{:.4f} ± {:.4f}'.format(
                        final_average_output_dict['false_positives_reg_analysis']['total_entropy_mean'],
                        final_average_output_dict['false_positives_reg_analysis']['total_entropy_mean_std']),
                    '-'])

            table.add_row(["False Negatives:",
                           num_false_negatives,
                           '-',
                           '-',
                           '-',
                           '-'])
            print(table)

            text_file_name = os.path.join(
                inference_output_dir,
                'probabilistic_scoring_res_{}_{}_{}.txt'.format(
                    iou_min,
                    iou_correct,
                    min_allowed_score))

            with open(text_file_name, "w") as text_file:
                print(table, file=text_file)

        dictionary_file_name = os.path.join(
            inference_output_dir, 'probabilistic_scoring_res_{}_{}_{}.pkl'.format(
                iou_min, iou_correct, min_allowed_score))

        with open(dictionary_file_name, "wb") as pickle_file:
            pickle.dump(final_accumulated_output_dict, pickle_file)


if __name__ == "__main__":
    # Create arg parser
    arg_parser = setup_arg_parser()

    args = arg_parser.parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
