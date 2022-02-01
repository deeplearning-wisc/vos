import numpy as np
import os
import pickle

from prettytable import PrettyTable

# Detectron imports
from detectron2.engine import launch

# Project imports
from core.setup import setup_config, setup_arg_parser
from offline_evaluation import compute_probabilistic_metrics, compute_calibration_errors
from inference.inference_utils import get_inference_output_dir


def main(args):
    cfg = setup_config(args,
                       random_seed=args.random_seed,
                       is_testing=True)

    inference_output_dir = get_inference_output_dir(
        cfg['OUTPUT_DIR'],
        args.test_dataset,
        args.inference_config,
        args.image_corruption_level)

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

    iou_thresholds = np.arange(0.5, 1.0, 0.05).round(2)

    probabilistic_detection_dicts = []
    calibration_dicts = []

    for iou_correct in iou_thresholds:
        print("Processing detections at {} iou threshold...".format(iou_correct))
        probabilistic_scores_file_name = os.path.join(
            inference_output_dir, 'probabilistic_scoring_res_{}_{}_{}.pkl'.format(
                args.iou_min, iou_correct, min_allowed_score))
        calibration_file_name = os.path.join(
            inference_output_dir, 'calibration_errors_res_{}_{}_{}.pkl'.format(
                args.iou_min, iou_correct, min_allowed_score))

        try:
            with open(probabilistic_scores_file_name, "rb") as f:
                probabilistic_scores = pickle.load(f)
        except FileNotFoundError:
            compute_probabilistic_metrics.main(
                args, cfg, iou_correct=iou_correct, print_results=False)
            with open(probabilistic_scores_file_name, "rb") as f:
                probabilistic_scores = pickle.load(f)

        try:
            with open(calibration_file_name, "rb") as f:
                calibration_errors = pickle.load(f)
        except FileNotFoundError:
            compute_calibration_errors.main(
                args, cfg, iou_correct=iou_correct, print_results=False)
            with open(calibration_file_name, "rb") as f:
                calibration_errors = pickle.load(f)

        probabilistic_detection_dicts.append(probabilistic_scores)
        calibration_dicts.append(calibration_errors)

    probabilistic_detection_final_dict = {
        key: {} for key in probabilistic_detection_dicts[0].keys()}
    for key in probabilistic_detection_dicts[0].keys():
        for key_l2 in probabilistic_detection_dicts[0][key].keys():
            accumulated_values = [
                probabilistic_detection_dicts[i][key][key_l2] for i in range(
                    len(probabilistic_detection_dicts))]
            probabilistic_detection_final_dict[key].update(
                {key_l2: np.nanmean(np.array(accumulated_values), 0)})

    calibration_final_dict = {key: None for key in calibration_dicts[0].keys()}
    for key in calibration_dicts[0].keys():
        accumulated_values = [
            calibration_dicts[i][key] for i in range(
                len(calibration_dicts))]
        calibration_final_dict[key] = np.nanmean(
            np.array(accumulated_values), 0)

    dictionary_file_name = os.path.join(
        inference_output_dir,
        'probabilistic_scoring_res_averaged_{}.pkl'.format(min_allowed_score))
    with open(dictionary_file_name, "wb") as pickle_file:
        pickle.dump(probabilistic_detection_final_dict, pickle_file)

    dictionary_file_name = os.path.join(
        inference_output_dir, 'calibration_res_averaged_{}.pkl'.format(
            min_allowed_score))
    with open(dictionary_file_name, "wb") as pickle_file:
        pickle.dump(calibration_final_dict, pickle_file)

    # Summarize and print all
    table = PrettyTable()
    table.field_names = (['Output Type',
                          'Cls Ignorance Score',
                          'Cls Brier/Probability Score',
                          'Reg Ignorance Score',
                          'Reg Energy Score'])

    table.add_row(
        [
            "True Positives:",
            '{:.4f}'.format(
                np.nanmean(probabilistic_detection_final_dict['true_positives_cls_analysis']['ignorance_score_mean'])),
            '{:.4f}'.format(
                np.nanmean(probabilistic_detection_final_dict['true_positives_cls_analysis']['brier_score_mean'])),
            '{:.4f}'.format(
                np.nanmean(probabilistic_detection_final_dict['true_positives_reg_analysis']['ignorance_score_mean'])),
            '{:.4f}'.format(
                np.nanmean(probabilistic_detection_final_dict['true_positives_reg_analysis']['energy_score_mean']))])
    table.add_row(
        [
            "Duplicates:",
            '{:.4f}'.format(
                np.nanmean(probabilistic_detection_final_dict['duplicates_cls_analysis']['ignorance_score_mean'])),
            '{:.4f}'.format(
                np.nanmean(probabilistic_detection_final_dict['duplicates_cls_analysis']['brier_score_mean'])),
            '{:.4f}'.format(
                np.nanmean(probabilistic_detection_final_dict['duplicates_reg_analysis']['ignorance_score_mean'])),
            '{:.4f}'.format(
                np.nanmean(probabilistic_detection_final_dict['duplicates_reg_analysis']['energy_score_mean']))])
    table.add_row(
        [
            "Localization Errors:",
            '{:.4f}'.format(
                np.nanmean(probabilistic_detection_final_dict['localization_errors_cls_analysis']['ignorance_score_mean'])),
            '{:.4f}'.format(
                np.nanmean(probabilistic_detection_final_dict['localization_errors_cls_analysis']['brier_score_mean'])),
            '{:.4f}'.format(
                np.nanmean(probabilistic_detection_final_dict['localization_errors_reg_analysis']['ignorance_score_mean'])),
            '{:.4f}'.format(
                np.nanmean(probabilistic_detection_final_dict['localization_errors_reg_analysis']['energy_score_mean']))])
    table.add_row(
        [
            "False Positives:",
            '{:.4f}'.format(
                np.nanmean(probabilistic_detection_final_dict['false_positives_cls_analysis']['ignorance_score_mean'])),
            '{:.4f}'.format(
                np.nanmean(probabilistic_detection_final_dict['false_positives_cls_analysis']['brier_score_mean'])),
            '{:.4f}'.format(
                np.nanmean(probabilistic_detection_final_dict['false_positives_reg_analysis']['total_entropy_mean'])),
            '{:.4f}'.format(
                np.nanmean(probabilistic_detection_final_dict['false_positives_reg_analysis']['fp_energy_score_mean']))])

    print(table)
    text_file_name = os.path.join(
        inference_output_dir,
        'probabilistic_scoring_res_averaged_{}.txt'.format(
            min_allowed_score))

    with open(text_file_name, "w") as text_file:
        print(table, file=text_file)

    table = PrettyTable()
    table.field_names = (['Cls Marginal Calibration Error',
                          'Reg Expected Calibration Error',
                          'Reg Maximum Calibration Error'])

    table.add_row(
        [
            '{:.4f}'.format(
                calibration_final_dict['cls_marginal_calibration_error']), '{:.4f}'.format(
                calibration_final_dict['reg_expected_calibration_error']), '{:.4f}'.format(
                    calibration_final_dict['reg_maximum_calibration_error'])])

    text_file_name = os.path.join(
        inference_output_dir,
        'calibration_res_averaged_{}.txt'.format(
            min_allowed_score))

    with open(text_file_name, "w") as text_file:
        print(table, file=text_file)

    print(table)


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
