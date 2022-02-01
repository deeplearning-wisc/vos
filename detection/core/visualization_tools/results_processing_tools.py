import glob
import itertools
import numpy as np
import os
import pickle
import torch

from collections import defaultdict

# Project imports
from core.setup import setup_config, setup_arg_parser
from inference.inference_utils import get_inference_output_dir


def get_clean_results_dict(config_names,
                           configs_list,
                           inference_configs_list):

    # Level 0 is coco validation set with no corruption, level 10 is open
    # images, level 11 is open images ood
    image_corruption_levels = [0, 1, 3, 5, 10, 11]

    test_dataset_coco = "coco_2017_custom_val"
    test_dataset_open_images = "openimages_val"
    test_dataset_open_images_odd = "openimages_odd_val"

    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    # Initiate dataframe dict
    res_dict_clean = defaultdict(lambda: defaultdict(list))

    for config_name, config, inference_config_name in zip(
            config_names, configs_list, inference_configs_list):
        # Setup config
        args.config_file = config
        args.inference_config = inference_config_name
        args.test_dataset = test_dataset_coco
        cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)
        cfg.defrost()

        # Read coco dataset results
        cfg.ACTUAL_TEST_DATASET = args.test_dataset

        for image_corruption_level in image_corruption_levels:
            # Build path to gt instances and inference output
            args.image_corruption_level = image_corruption_level

            if image_corruption_level == 0:
                image_corruption_level = 'Val'
            elif image_corruption_level == 10:
                image_corruption_level = 'OpenIm'
            elif image_corruption_level == 11:
                image_corruption_level = 'OpenIm OOD'
            else:
                image_corruption_level = 'C' + str(image_corruption_level)
            if 'OpenIm' not in image_corruption_level:
                inference_output_dir = get_inference_output_dir(
                    cfg['OUTPUT_DIR'],
                    args.test_dataset,
                    args.inference_config,
                    args.image_corruption_level)

                dictionary_file_name = glob.glob(
                    os.path.join(
                        inference_output_dir,
                        'probabilistic_scoring_res_averaged_*.pkl'))[0]
            else:
                args.image_corruption_level = 0
                args.test_dataset = test_dataset_open_images if image_corruption_level == 'OpenIm' else test_dataset_open_images_odd
                inference_output_dir = get_inference_output_dir(
                    cfg['OUTPUT_DIR'],
                    args.test_dataset,
                    args.inference_config,
                    args.image_corruption_level)
                prob_dict_name = 'probabilistic_scoring_res_averaged_*.pkl' if image_corruption_level == 'OpenIm' else 'probabilistic_scoring_res_odd_*.pkl'
                dictionary_file_name = glob.glob(
                    os.path.join(
                        inference_output_dir,
                        prob_dict_name))[0]

            with open(dictionary_file_name, "rb") as pickle_file:
                res_dict = pickle.load(pickle_file)

                if image_corruption_level != 'OpenIm OOD':
                    # True Positives Results
                    res_dict_clean['True Positives']['Negative Log Likelihood (Classification)'].extend(
                        res_dict['true_positives_cls_analysis']['ignorance_score_mean'])
                    res_dict_clean['True Positives']['Brier Score'].extend(
                        res_dict['true_positives_cls_analysis']['brier_score_mean'])
                    res_dict_clean['True Positives']['Negative Log Likelihood (Regression)'].extend(
                        res_dict['true_positives_reg_analysis']['ignorance_score_mean'])
                    res_dict_clean['True Positives']['Mean Squared Error'].extend(
                        res_dict['true_positives_reg_analysis']['mean_squared_error'])
                    res_dict_clean['True Positives']['Energy Score'].extend(
                        res_dict['true_positives_reg_analysis']['energy_score_mean'])
                    res_dict_clean['True Positives']['Image Corruption Level'].extend(
                        [image_corruption_level] *
                        res_dict['true_positives_reg_analysis']['energy_score_mean'].shape[0])
                    res_dict_clean['True Positives']['Method Name'].extend(
                        [config_name] * res_dict['true_positives_reg_analysis']['energy_score_mean'].shape[0])

                    # Duplicates Results
                    res_dict_clean['Duplicates']['Negative Log Likelihood (Classification)'].extend(
                        res_dict['duplicates_cls_analysis']['ignorance_score_mean'])
                    res_dict_clean['Duplicates']['Brier Score'].extend(
                        res_dict['duplicates_cls_analysis']['brier_score_mean'])
                    res_dict_clean['Duplicates']['Negative Log Likelihood (Regression)'].extend(
                        res_dict['duplicates_reg_analysis']['ignorance_score_mean'])
                    res_dict_clean['Duplicates']['Mean Squared Error'].extend(
                        res_dict['duplicates_reg_analysis']['mean_squared_error'])
                    res_dict_clean['Duplicates']['Energy Score'].extend(
                        res_dict['duplicates_reg_analysis']['energy_score_mean'])
                    res_dict_clean['Duplicates']['Image Corruption Level'].extend(
                        [image_corruption_level] *
                        res_dict['duplicates_reg_analysis']['energy_score_mean'].shape[0])
                    res_dict_clean['Duplicates']['Method Name'].extend(
                        [config_name] * res_dict['duplicates_reg_analysis']['energy_score_mean'].shape[0])

                    # Localization Error Results
                    res_dict_clean['Localization Errors']['Negative Log Likelihood (Classification)'].extend(
                        res_dict['localization_errors_cls_analysis']['ignorance_score_mean'])
                    res_dict_clean['Localization Errors']['Brier Score'].extend(
                        res_dict['localization_errors_cls_analysis']['brier_score_mean'])
                    res_dict_clean['Localization Errors']['Negative Log Likelihood (Regression)'].extend(
                        res_dict['localization_errors_reg_analysis']['ignorance_score_mean'])
                    res_dict_clean['Localization Errors']['Mean Squared Error'].extend(
                        res_dict['localization_errors_reg_analysis']['mean_squared_error'])
                    res_dict_clean['Localization Errors']['Energy Score'].extend(
                        res_dict['localization_errors_reg_analysis']['energy_score_mean'])
                    res_dict_clean['Localization Errors']['Image Corruption Level'].extend(
                        [image_corruption_level] *
                        res_dict['localization_errors_reg_analysis']['energy_score_mean'].shape[0])
                    res_dict_clean['Localization Errors']['Method Name'].extend(
                        [config_name] *
                        res_dict['localization_errors_reg_analysis']['energy_score_mean'].shape[0])

                    # False Positives Results
                    res_dict_clean['False Positives']['Negative Log Likelihood (Classification)'].extend(
                        res_dict['false_positives_cls_analysis']['ignorance_score_mean'])
                    res_dict_clean['False Positives']['Brier Score'].extend(
                        res_dict['false_positives_cls_analysis']['brier_score_mean'])
                    res_dict_clean['False Positives']['Entropy'].extend(
                        res_dict['false_positives_reg_analysis']['total_entropy_mean'])
                    res_dict_clean['False Positives']['Image Corruption Level'].extend(
                        [image_corruption_level] *
                        res_dict['false_positives_reg_analysis']['total_entropy_mean'].shape[0])
                    res_dict_clean['False Positives']['Method Name'].extend(
                        [config_name] *
                        res_dict['false_positives_reg_analysis']['total_entropy_mean'].shape[0])
                else:
                    # False Positives Results
                    res_dict_clean['False Positives']['Negative Log Likelihood (Classification)'].append(
                        res_dict['ignorance_score_mean'])
                    res_dict_clean['False Positives']['Brier Score'].append(
                        res_dict['brier_score_mean'])
                    res_dict_clean['False Positives']['Entropy'].append(
                        res_dict['total_entropy_mean'])
                    res_dict_clean['False Positives']['Image Corruption Level'].append(
                        image_corruption_level)
                    res_dict_clean['False Positives']['Method Name'].append(
                        config_name)
    return res_dict_clean


def get_mAP_results(config_names,
                    configs_list,
                    inference_configs_list):
    # Level 0 is coco validation set with no corruption, level 10 is open
    # images, level 11 is open images ood
    image_corruption_levels = [0, 1, 2, 3, 4, 5, 10]

    test_dataset_coco = "coco_2017_custom_val"
    test_dataset_open_images = "openimages_val"

    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    # Initiate dataframe dict
    mAP_results = defaultdict(list)

    for config_name, config, inference_config_name in zip(
            config_names, configs_list, inference_configs_list):
        # Setup config
        args.config_file = config
        args.inference_config = inference_config_name
        args.test_dataset = test_dataset_coco
        cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)
        cfg.defrost()

        # Read coco dataset results
        cfg.ACTUAL_TEST_DATASET = args.test_dataset

        for image_corruption_level in image_corruption_levels:
            # Build path to gt instances and inference output
            args.image_corruption_level = image_corruption_level
            if image_corruption_level == 0:
                image_corruption_level = 'Val'
            elif image_corruption_level == 10:
                image_corruption_level = 'OpenIm'
            else:
                image_corruption_level = 'C' + str(image_corruption_level)

            if 'OpenIm' not in image_corruption_level:
                inference_output_dir = get_inference_output_dir(
                    cfg['OUTPUT_DIR'],
                    args.test_dataset,
                    args.inference_config,
                    args.image_corruption_level)
            else:
                args.image_corruption_level = 0
                args.test_dataset = test_dataset_open_images
                inference_output_dir = get_inference_output_dir(
                    cfg['OUTPUT_DIR'],
                    args.test_dataset,
                    args.inference_config,
                    args.image_corruption_level)

            text_file_name = glob.glob(
                os.path.join(
                    inference_output_dir,
                    'mAP_res.txt'))[0]
            with open(text_file_name, "r") as f:
                mAP = f.read().strip('][\n').split(', ')[0]
                mAP = float(mAP) * 100

            mAP_results['Method Name'].append(config_name)
            mAP_results['Image Corruption Level'].append(
                image_corruption_level)
            mAP_results['mAP'].append(mAP)

    return mAP_results


def get_matched_results_dicts(config_names,
                              configs_list,
                              inference_configs_list,
                              iou_min=0.1,
                              iou_correct=0.5):

    # Level 0 is coco validation set with no corruption, level 10 is open
    # images, level 11 is open images ood
    image_corruption_levels = [0, 10, 11]

    test_dataset_coco = "coco_2017_custom_val"
    test_dataset_open_images = "openimages_val"
    test_dataset_open_images_odd = "openimages_odd_val"

    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    # Initiate dataframe dict
    res_dict_clean = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for config_name, config, inference_config_name in zip(
            config_names, configs_list, inference_configs_list):
        # Setup config
        args.config_file = config
        args.inference_config = inference_config_name
        args.test_dataset = test_dataset_coco
        cfg = setup_config(args, random_seed=args.random_seed, is_testing=True)
        cfg.defrost()

        # Read coco dataset results
        cfg.ACTUAL_TEST_DATASET = args.test_dataset

        for image_corruption_level in image_corruption_levels:
            # Build path to gt instances and inference output
            args.image_corruption_level = image_corruption_level

            if image_corruption_level == 0:
                image_corruption_level = 'Val'
            elif image_corruption_level == 10:
                image_corruption_level = 'OpenIm'
            elif image_corruption_level == 11:
                image_corruption_level = 'OpenIm OOD'
            else:
                image_corruption_level = 'C' + str(image_corruption_level)
            if 'OpenIm' not in image_corruption_level:
                inference_output_dir = get_inference_output_dir(
                    cfg['OUTPUT_DIR'],
                    args.test_dataset,
                    args.inference_config,
                    args.image_corruption_level)

                # Get matched results by either generating them or loading from
                # file.

                dictionary_file_name = glob.glob(
                    os.path.join(
                        inference_output_dir,
                        "matched_results_{}_{}_*.pth".format(
                            iou_min,
                            iou_correct)))[0]

                matched_results = torch.load(
                    dictionary_file_name, map_location='cuda')
            elif image_corruption_level == 'OpenIm':
                args.image_corruption_level = 0
                args.test_dataset = test_dataset_open_images if image_corruption_level == 'OpenIm' else test_dataset_open_images_odd
                inference_output_dir = get_inference_output_dir(
                    cfg['OUTPUT_DIR'],
                    args.test_dataset,
                    args.inference_config,
                    args.image_corruption_level)
                dictionary_file_name = glob.glob(
                    os.path.join(
                        inference_output_dir,
                        "matched_results_{}_{}_*.pth".format(
                            iou_min,
                            iou_correct)))[0]
                matched_results = torch.load(
                    dictionary_file_name, map_location='cuda')
            else:
                args.image_corruption_level = 0
                args.test_dataset = test_dataset_open_images if image_corruption_level == 'OpenIm' else test_dataset_open_images_odd
                inference_output_dir = get_inference_output_dir(
                    cfg['OUTPUT_DIR'],
                    args.test_dataset,
                    args.inference_config,
                    args.image_corruption_level)
                dictionary_file_name = glob.glob(
                    os.path.join(
                        inference_output_dir,
                        "preprocessed_predicted_instances_odd_*.pth"))[0]
                preprocessed_predicted_instances = torch.load(
                    dictionary_file_name, map_location='cuda')

                predicted_boxes = preprocessed_predicted_instances['predicted_boxes']
                predicted_cov_mats = preprocessed_predicted_instances['predicted_covar_mats']
                predicted_cls_probs = preprocessed_predicted_instances['predicted_cls_probs']

                predicted_boxes = list(itertools.chain.from_iterable(
                    [predicted_boxes[key] for key in predicted_boxes.keys()]))
                predicted_cov_mats = list(itertools.chain.from_iterable(
                    [predicted_cov_mats[key] for key in predicted_cov_mats.keys()]))
                predicted_cls_probs = list(itertools.chain.from_iterable(
                    [predicted_cls_probs[key] for key in predicted_cls_probs.keys()]))

                predicted_boxes = torch.stack(
                    predicted_boxes, 1).transpose(
                    0, 1)
                predicted_cov_mats = torch.stack(
                    predicted_cov_mats, 1).transpose(0, 1)
                predicted_cls_probs = torch.stack(
                    predicted_cls_probs,
                    1).transpose(
                    0,
                    1)
                matched_results = {
                    'predicted_box_means': predicted_boxes,
                    'predicted_box_covariances': predicted_cov_mats,
                    'predicted_cls_probs': predicted_cls_probs}

            if image_corruption_level != 'OpenIm OOD':
                all_results_means = torch.cat(
                    (matched_results['true_positives']['predicted_box_means'],
                     matched_results['localization_errors']['predicted_box_means'],
                     matched_results['duplicates']['predicted_box_means'],
                     matched_results['false_positives']['predicted_box_means']))

                all_results_covs = torch.cat(
                    (matched_results['true_positives']['predicted_box_covariances'],
                     matched_results['localization_errors']['predicted_box_covariances'],
                     matched_results['duplicates']['predicted_box_covariances'],
                     matched_results['false_positives']['predicted_box_covariances']))

                all_gt_means = torch.cat(
                    (matched_results['true_positives']['gt_box_means'],
                     matched_results['localization_errors']['gt_box_means'],
                     matched_results['duplicates']['gt_box_means'],
                     matched_results['false_positives']['predicted_box_means']*np.NaN))

                predicted_multivariate_normal_dists = torch.distributions.multivariate_normal.MultivariateNormal(
                    all_results_means.to('cpu'),
                    all_results_covs.to('cpu') +
                    1e-2 *
                    torch.eye(all_results_covs.shape[2]).to('cpu'))
                predicted_multivariate_normal_dists.loc = predicted_multivariate_normal_dists.loc.to(
                    'cuda')
                predicted_multivariate_normal_dists.scale_tril = predicted_multivariate_normal_dists.scale_tril.to(
                    'cuda')
                predicted_multivariate_normal_dists._unbroadcasted_scale_tril = predicted_multivariate_normal_dists._unbroadcasted_scale_tril.to(
                    'cuda')
                predicted_multivariate_normal_dists.covariance_matrix = predicted_multivariate_normal_dists.covariance_matrix.to(
                    'cuda')
                predicted_multivariate_normal_dists.precision_matrix = predicted_multivariate_normal_dists.precision_matrix.to(
                    'cuda')
                all_entropy = predicted_multivariate_normal_dists.entropy()

                all_log_prob = -predicted_multivariate_normal_dists.log_prob(all_gt_means)
                # Energy Score.
                sample_set = predicted_multivariate_normal_dists.sample((3,)).to('cuda')
                sample_set_1 = sample_set[:-1]
                sample_set_2 = sample_set[1:]

                energy_score = torch.norm(
                    (sample_set_1 - all_gt_means),
                    dim=2).mean(0) - 0.5 * torch.norm(
                    (sample_set_1 - sample_set_2),
                    dim=2).mean(0)

                mse_loss = torch.nn.MSELoss(reduction='none')
                mse = mse_loss(all_gt_means, all_results_means).mean(1)

                res_dict_clean[config_name][image_corruption_level]['Entropy'].extend(
                    all_entropy.cpu().numpy())

                res_dict_clean[config_name][image_corruption_level]['MSE'].extend(
                    mse.cpu().numpy())
                res_dict_clean[config_name][image_corruption_level]['NLL'].extend(
                    all_log_prob.cpu().numpy())
                res_dict_clean[config_name][image_corruption_level]['ED'].extend(
                    energy_score.cpu().numpy())

                res_dict_clean[config_name][image_corruption_level]['IOU With GT'].extend(torch.cat(
                        (matched_results['true_positives']['iou_with_ground_truth'],
                         matched_results['localization_errors']['iou_with_ground_truth'][:, 0],
                         matched_results['duplicates']['iou_with_ground_truth'],
                         torch.zeros(
                            matched_results['false_positives']['predicted_box_means'].shape[0]).to('cuda')*np.NaN)).cpu().numpy())

                predicted_multivariate_normal_dists = torch.distributions.multivariate_normal.MultivariateNormal(
                    matched_results['false_positives']['predicted_box_means'].to('cpu'),
                    matched_results['false_positives']['predicted_box_covariances'].to('cpu') +
                    1e-2 *
                    torch.eye(matched_results['false_positives']['predicted_box_covariances'].shape[2]).to('cpu'))
                predicted_multivariate_normal_dists.loc = predicted_multivariate_normal_dists.loc.to(
                    'cuda')
                predicted_multivariate_normal_dists.scale_tril = predicted_multivariate_normal_dists.scale_tril.to(
                    'cuda')
                predicted_multivariate_normal_dists._unbroadcasted_scale_tril = predicted_multivariate_normal_dists._unbroadcasted_scale_tril.to(
                    'cuda')
                predicted_multivariate_normal_dists.covariance_matrix = predicted_multivariate_normal_dists.covariance_matrix.to(
                    'cuda')
                predicted_multivariate_normal_dists.precision_matrix = predicted_multivariate_normal_dists.precision_matrix.to(
                    'cuda')
                FP_Entropy = predicted_multivariate_normal_dists.entropy()
                res_dict_clean[config_name][image_corruption_level]['FP_Entropy'].extend(
                    FP_Entropy.cpu().numpy())

                predicted_cat_dists_fp = matched_results['false_positives']['predicted_cls_probs']

                if predicted_cat_dists_fp.shape[1] == 80:
                    predicted_cat_dists_fp, _ = predicted_cat_dists_fp.max(dim=1)
                    predicted_cat_dists_fp = 1-predicted_cat_dists_fp
                    predicted_categorical_dists = torch.distributions.Bernoulli(
                        probs=predicted_cat_dists_fp)
                else:
                    predicted_categorical_dists = torch.distributions.Categorical(
                        probs=matched_results['false_positives']['predicted_cls_probs'])

                all_pred_ent = predicted_categorical_dists.entropy()
                res_dict_clean[config_name][image_corruption_level]['Cat_Entropy'].extend(
                    all_pred_ent.cpu().numpy())

                if image_corruption_level == 'OpenIm':
                    res_dict_clean[config_name][image_corruption_level]['Truncated'].extend(
                        torch.cat(
                            (matched_results['true_positives']['is_truncated'],
                             matched_results['localization_errors']['is_truncated'],
                             matched_results['duplicates']['is_truncated'],
                             torch.full((
                                matched_results['false_positives']['predicted_box_means'].shape[0],), -1, dtype=torch.float32).to('cuda')*np.NaN)).cpu().numpy())
                    res_dict_clean[config_name][image_corruption_level]['Occluded'].extend(
                        torch.cat(
                            (matched_results['true_positives']['is_occluded'],
                             matched_results['localization_errors']['is_occluded'],
                             matched_results['duplicates']['is_occluded'],
                             torch.full((
                                matched_results['false_positives']['predicted_box_means'].shape[0],), -1, dtype=torch.float32).to('cuda')*np.NaN)).cpu().numpy())
                else:
                    res_dict_clean[config_name][image_corruption_level]['Truncated'].extend(
                        torch.cat(
                            (torch.full((
                                matched_results['true_positives']['predicted_box_means'].shape[0],), -1, dtype=torch.float32).to('cuda')*np.NaN,
                             torch.full((
                                 matched_results['localization_errors']['predicted_box_means'].shape[0],), -1,
                                 dtype=torch.float32).to('cuda'),
                             torch.full((
                                 matched_results['duplicates']['predicted_box_means'].shape[0],), -1,
                                 dtype=torch.float32).to('cuda'),
                             torch.full((
                                matched_results['false_positives']['predicted_box_means'].shape[0],), -1, dtype=torch.float32).to('cuda')*np.NaN)).cpu().numpy())
                    res_dict_clean[config_name][image_corruption_level]['Occluded'].extend(
                        torch.cat(
                            (torch.full((
                                matched_results['true_positives']['predicted_box_means'].shape[0],), -1, dtype=torch.float32).to('cuda')*np.NaN,
                             torch.full((
                                 matched_results['localization_errors']['predicted_box_means'].shape[0],), -1,
                                 dtype=torch.float32).to('cuda')*np.NaN,
                             torch.full((
                                 matched_results['duplicates']['predicted_box_means'].shape[0],), -1,
                                 dtype=torch.float32).to('cuda')*np.NaN,
                             torch.full((
                                matched_results['false_positives']['predicted_box_means'].shape[0],), -1, dtype=torch.float32).to('cuda')*np.NaN)).cpu().numpy())
            else:
                predicted_multivariate_normal_dists = torch.distributions.multivariate_normal.MultivariateNormal(
                    matched_results['predicted_box_means'].to('cpu'),
                    matched_results['predicted_box_covariances'].to('cpu') +
                    1e-2 *
                    torch.eye(matched_results['predicted_box_covariances'].shape[2]).to('cpu'))
                predicted_multivariate_normal_dists.loc = predicted_multivariate_normal_dists.loc.to(
                    'cuda')
                predicted_multivariate_normal_dists.scale_tril = predicted_multivariate_normal_dists.scale_tril.to(
                    'cuda')
                predicted_multivariate_normal_dists._unbroadcasted_scale_tril = predicted_multivariate_normal_dists._unbroadcasted_scale_tril.to(
                    'cuda')
                predicted_multivariate_normal_dists.covariance_matrix = predicted_multivariate_normal_dists.covariance_matrix.to(
                    'cuda')
                predicted_multivariate_normal_dists.precision_matrix = predicted_multivariate_normal_dists.precision_matrix.to(
                    'cuda')
                all_entropy = predicted_multivariate_normal_dists.entropy()
                res_dict_clean[config_name][image_corruption_level]['FP_Entropy'].extend(
                    all_entropy.cpu().numpy())
                res_dict_clean[config_name][image_corruption_level]['IOU With GT'].extend(torch.zeros(
                    matched_results['predicted_box_means'].shape[0]).cpu().numpy())
                res_dict_clean[config_name][image_corruption_level]['Truncated'].extend(torch.full((
                    matched_results['predicted_box_means'].shape[0],), -1, dtype=torch.float32).cpu().numpy()*np.NaN)
                res_dict_clean[config_name][image_corruption_level]['Occluded'].extend(torch.full((
                    matched_results['predicted_box_means'].shape[0],), -1, dtype=torch.float32).cpu().numpy()*np.NaN)

                all_results_cat = matched_results['predicted_cls_probs']
                if all_results_cat.shape[1] == 80:
                    predicted_cat_dists_fp, _ = all_results_cat.max(dim=1)
                    predicted_cat_dists_fp = 1-predicted_cat_dists_fp
                    predicted_categorical_dists = torch.distributions.Bernoulli(
                        probs=predicted_cat_dists_fp)
                else:
                    predicted_categorical_dists = torch.distributions.Categorical(
                        probs=all_results_cat)

                all_pred_ent = predicted_categorical_dists.entropy()
                res_dict_clean[config_name][image_corruption_level]['Cat_Entropy'].extend(
                    all_pred_ent.cpu().numpy())

    return res_dict_clean


def mean_reject_outliers(x, outlierConstant=1.5):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)

    result = a[np.where((a >= quartileSet[0]) & (a <= quartileSet[1]))]
    return np.nanmean(result)
