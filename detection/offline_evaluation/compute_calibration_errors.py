import calibration as cal
import os
import pickle
import torch

from prettytable import PrettyTable

# Detectron imports
from detectron2.data import MetadataCatalog
from detectron2.engine import launch

# Project imports
from core.evaluation_tools import evaluation_utils
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
                matched_results[matched_results_key]['gt_cat_idxs'] = gt_converted_cat_idxs
            if 'predicted_cls_probs' in matched_results[matched_results_key].keys(
            ):
                if cfg.MODEL.META_ARCHITECTURE == 'ProbabilisticRetinaNet':
                    # For false positives, the correct category is background. For retinanet, since no explicit
                    # background category is available, this value is computed as 1.0 - score of the predicted
                    # category.
                    predicted_class_probs, predicted_cat_idxs = matched_results[matched_results_key][
                        'predicted_cls_probs'].max(
                        1)
                    matched_results[matched_results_key]['output_logits'] = predicted_class_probs
                else:
                    predicted_class_probs, predicted_cat_idxs = matched_results[
                        matched_results_key]['predicted_cls_probs'][:, :-1].max(1)

                matched_results[matched_results_key]['predicted_cat_idxs'] = predicted_cat_idxs

        # Load the different detection partitions
        true_positives = matched_results['true_positives']
        duplicates = matched_results['duplicates']
        localization_errors = matched_results['localization_errors']
        false_positives = matched_results['false_positives']

        reg_maximum_calibration_error_list = []
        reg_expected_calibration_error_list = []
        if cfg.MODEL.META_ARCHITECTURE == 'ProbabilisticRetinaNet':
            all_predicted_scores = torch.cat(
                (true_positives['predicted_cls_probs'].flatten(),
                 duplicates['predicted_cls_probs'].flatten(),
                 localization_errors['predicted_cls_probs'].flatten(),
                 false_positives['predicted_cls_probs'].flatten()),
                0)
            all_gt_scores = torch.cat(
                (torch.nn.functional.one_hot(
                    true_positives['gt_cat_idxs'],
                    true_positives['predicted_cls_probs'].shape[1]).flatten().to(device),
                 torch.nn.functional.one_hot(
                     duplicates['gt_cat_idxs'],
                     duplicates['predicted_cls_probs'].shape[1]).flatten().to(device),
                 torch.zeros_like(
                     localization_errors['predicted_cls_probs'].type(
                         torch.LongTensor).flatten()).to(device),
                 torch.zeros_like(
                     false_positives['predicted_cls_probs'].type(
                         torch.LongTensor).flatten()).to(device)),
                0)
        else:
            # For RCNN based networks, a background category is
            # explicitly available.
            all_predicted_scores = torch.cat(
                (true_positives['predicted_cls_probs'],
                 duplicates['predicted_cls_probs'],
                 localization_errors['predicted_cls_probs'],
                 false_positives['predicted_cls_probs']),
                0)
            all_gt_scores = torch.cat(
                (true_positives['gt_cat_idxs'],
                 duplicates['gt_cat_idxs'],
                 torch.ones_like(
                     localization_errors['predicted_cls_probs'][:, 0]).fill_(80.0).type(
                     torch.LongTensor).to(device),
                 torch.ones_like(
                     false_positives['predicted_cls_probs'][:, 0]).fill_(80.0).type(
                     torch.LongTensor).to(device)), 0)

        # Compute classification calibration error using calibration
        # library
        cls_marginal_calibration_error = cal.get_calibration_error(
            all_predicted_scores.cpu().numpy(), all_gt_scores.cpu().numpy())

        for class_idx in cat_mapping_dict.values():
            true_positives_valid_idxs = true_positives['gt_converted_cat_idxs'] == class_idx
            localization_errors_valid_idxs = localization_errors['gt_converted_cat_idxs'] == class_idx
            duplicates_valid_idxs = duplicates['gt_converted_cat_idxs'] == class_idx

            # Compute regression calibration errors. False negatives cant be evaluated since
            # those do not have ground truth.
            all_predicted_means = torch.cat(
                (true_positives['predicted_box_means'][true_positives_valid_idxs],
                 duplicates['predicted_box_means'][duplicates_valid_idxs],
                 localization_errors['predicted_box_means'][localization_errors_valid_idxs]),
                0)

            all_predicted_covariances = torch.cat(
                (true_positives['predicted_box_covariances'][true_positives_valid_idxs],
                 duplicates['predicted_box_covariances'][duplicates_valid_idxs],
                 localization_errors['predicted_box_covariances'][localization_errors_valid_idxs]),
                0)

            all_predicted_gt = torch.cat(
                (true_positives['gt_box_means'][true_positives_valid_idxs],
                 duplicates['gt_box_means'][duplicates_valid_idxs],
                 localization_errors['gt_box_means'][localization_errors_valid_idxs]),
                0)

            all_predicted_covariances = torch.diagonal(
                all_predicted_covariances, dim1=1, dim2=2)

            # The assumption of uncorrelated components is not accurate, especially when estimating full
            # covariance matrices. However, using scipy to compute multivariate cdfs is very very
            # time consuming for such large amounts of data.
            reg_maximum_calibration_error = []
            reg_expected_calibration_error = []

            # Regression calibration is computed for every box dimension
            # separately, and averaged after.
            for box_dim in range(all_predicted_gt.shape[1]):
                all_predicted_means_current_dim = all_predicted_means[:, box_dim]
                all_predicted_gt_current_dim = all_predicted_gt[:, box_dim]
                all_predicted_covariances_current_dim = all_predicted_covariances[:, box_dim]
                normal_dists = torch.distributions.Normal(
                    all_predicted_means_current_dim,
                    scale=torch.sqrt(all_predicted_covariances_current_dim))
                all_predicted_scores = normal_dists.cdf(
                    all_predicted_gt_current_dim)

                reg_calibration_error = []
                histogram_bin_step_size = 1 / 15.0
                for i in torch.arange(
                        0.0,
                        1.0 - histogram_bin_step_size,
                        histogram_bin_step_size):
                    # Get number of elements in bin
                    elements_in_bin = (
                        all_predicted_scores < (i + histogram_bin_step_size))
                    num_elems_in_bin_i = elements_in_bin.type(
                        torch.FloatTensor).to(device).sum()

                    # Compute calibration error from "Accurate uncertainties for deep
                    # learning using calibrated regression" paper.
                    reg_calibration_error.append(
                        (num_elems_in_bin_i / all_predicted_scores.shape[0] - (i + histogram_bin_step_size)) ** 2)

                calibration_error = torch.stack(
                    reg_calibration_error).to(device)
                reg_maximum_calibration_error.append(calibration_error.max())
                reg_expected_calibration_error.append(calibration_error.mean())

            reg_maximum_calibration_error_list.append(
                reg_maximum_calibration_error)
            reg_expected_calibration_error_list.append(
                reg_expected_calibration_error)

        # Summarize and print all
        reg_expected_calibration_error = torch.stack([torch.stack(
            reg, 0) for reg in reg_expected_calibration_error_list], 0)
        reg_expected_calibration_error = reg_expected_calibration_error[
            ~torch.isnan(reg_expected_calibration_error)].mean()

        reg_maximum_calibration_error = torch.stack([torch.stack(
            reg, 0) for reg in reg_maximum_calibration_error_list], 0)
        reg_maximum_calibration_error = reg_maximum_calibration_error[
            ~torch.isnan(reg_maximum_calibration_error)].mean()

        if print_results:
            table = PrettyTable()
            table.field_names = (['Cls Marginal Calibration Error',
                                  'Reg Expected Calibration Error',
                                  'Reg Maximum Calibration Error'])

            table.add_row([cls_marginal_calibration_error,
                           reg_expected_calibration_error.cpu().numpy().tolist(),
                           reg_maximum_calibration_error.cpu().numpy().tolist()])
            print(table)

            text_file_name = os.path.join(
                inference_output_dir,
                'calibration_errors_{}_{}_{}.txt'.format(
                    iou_min, iou_correct, min_allowed_score))

            with open(text_file_name, "w") as text_file:
                print([
                    cls_marginal_calibration_error,
                    reg_expected_calibration_error.cpu().numpy().tolist(),
                    reg_maximum_calibration_error.cpu().numpy().tolist()], file=text_file)

        dictionary_file_name = os.path.join(
            inference_output_dir, 'calibration_errors_res_{}_{}_{}.pkl'.format(
                iou_min, iou_correct, min_allowed_score))

        final_accumulated_output_dict = {
            'cls_marginal_calibration_error': cls_marginal_calibration_error,
            'reg_expected_calibration_error': reg_expected_calibration_error.cpu().numpy(),
            'reg_maximum_calibration_error': reg_maximum_calibration_error.cpu().numpy()}

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
