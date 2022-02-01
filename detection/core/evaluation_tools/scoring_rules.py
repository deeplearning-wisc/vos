import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sigmoid_compute_cls_scores(input_matches, valid_idxs):
    """
    Computes proper scoring rule for multilabel classification results provided by retinanet.

    Args:
        input_matches (dict): dictionary containing input matches
        valid_idxs (tensor): a tensor containing valid element idxs for per-class computation
    Returns:
        output_dict (dict): dictionary containing ignorance and brier score.
    """

    output_dict = {}
    num_forecasts = input_matches['predicted_cls_probs'][valid_idxs].shape[0]

    # Construct binary probability vectors. Essential for RetinaNet as it uses
    # multilabel and not multiclass formulation.

    predicted_class_probs = input_matches['predicted_score_of_gt_category'][valid_idxs]

    # If no valid idxs, do not perform computation
    if predicted_class_probs.shape[0] == 0:
        output_dict.update({'ignorance_score_mean': None,
                            'brier_score_mean': None})
        return output_dict
    predicted_multilabel_probs = torch.stack(
        [predicted_class_probs, 1.0 - predicted_class_probs], dim=1)

    correct_multilabel_probs = torch.stack(
        [torch.ones(num_forecasts),
         torch.zeros(num_forecasts)], dim=1).to(device)

    predicted_log_likelihood_of_correct_category = (
        -correct_multilabel_probs * torch.log(predicted_multilabel_probs)).sum(1)

    cls_ignorance_score_mean = predicted_log_likelihood_of_correct_category.mean()
    output_dict.update(
        {'ignorance_score_mean': cls_ignorance_score_mean.to(device).tolist()})

    # Classification Brier (Probability) Score
    predicted_brier_raw = ((predicted_multilabel_probs -
                            correct_multilabel_probs)**2).sum(1)
    cls_brier_score_mean = predicted_brier_raw.mean()
    output_dict.update(
        {'brier_score_mean': cls_brier_score_mean.to(device).tolist()})

    return output_dict


def softmax_compute_cls_scores(input_matches, valid_idxs):
    """
    Computes proper scoring rule for multiclass classification results provided by faster_rcnn.

    Args:
        input_matches (dict): dictionary containing input matches
        valid_idxs (tensor): a tensor containing valid element idxs for per-class computation
    Returns:
        output_dict (dict): dictionary containing ignorance and brier score.
    """
    output_dict = {}

    predicted_multilabel_probs = input_matches['predicted_cls_probs'][valid_idxs]
    if 'gt_cat_idxs' in input_matches.keys():
        correct_multilabel_probs = torch.nn.functional.one_hot(input_matches['gt_cat_idxs'][valid_idxs].type(
            torch.LongTensor), input_matches['predicted_cls_probs'][valid_idxs].shape[-1]).to(device)
    else:
        correct_multilabel_probs = torch.zeros_like(
            predicted_multilabel_probs).to(device)
        correct_multilabel_probs[:, -1] = 1.0

    if predicted_multilabel_probs.shape[0] == 0:
        output_dict.update({'ignorance_score_mean': None,
                            'brier_score_mean': None})
        return output_dict

    predicted_log_likelihood_of_correct_category = (
        -correct_multilabel_probs * torch.log(predicted_multilabel_probs)).sum(1)
    cls_ignorance_score_mean = predicted_log_likelihood_of_correct_category.mean()
    output_dict.update(
        {'ignorance_score_mean': cls_ignorance_score_mean.to(device).tolist()})

    # Classification Probability Score. Multiclass version of brier score.
    predicted_brier_raw = ((predicted_multilabel_probs -
                            correct_multilabel_probs)**2).sum(1)
    cls_brier_score_mean = predicted_brier_raw.mean()
    output_dict.update(
        {'brier_score_mean': cls_brier_score_mean.to(device).tolist()})

    return output_dict


def compute_reg_scores(input_matches, valid_idxs):
    """
    Computes proper scoring rule for regression results.

    Args:
        input_matches (dict): dictionary containing input matches
        valid_idxs (tensor): a tensor containing valid element idxs for per-class computation

    Returns:
        output_dict (dict): dictionary containing ignorance and energy scores.
    """
    output_dict = {}

    predicted_box_means = input_matches['predicted_box_means'][valid_idxs]
    predicted_box_covars = input_matches['predicted_box_covariances'][valid_idxs]
    gt_box_means = input_matches['gt_box_means'][valid_idxs]

    # If no valid idxs, do not perform computation
    if predicted_box_means.shape[0] == 0:
        output_dict.update({'ignorance_score_mean': None,
                            'mean_squared_error': None,
                            'energy_score_mean': None})
        return output_dict

    # Compute negative log likelihood
    # Note: Juggling between CPU and GPU is due to magma library unresolvable issue, where cuda illegal memory access
    # error is returned arbitrarily depending on the state of the GPU. This is only a problem for the
    # torch.distributions code.
    # Pytorch unresolved issue from 2019:
    # https://github.com/pytorch/pytorch/issues/21819
    predicted_multivariate_normal_dists = torch.distributions.multivariate_normal.MultivariateNormal(
        predicted_box_means.to('cpu'),
        predicted_box_covars.to('cpu') +
        1e-2 *
        torch.eye(
            predicted_box_covars.shape[2]).to('cpu'))
    predicted_multivariate_normal_dists.loc = predicted_multivariate_normal_dists.loc.to(
        device)
    predicted_multivariate_normal_dists.scale_tril = predicted_multivariate_normal_dists.scale_tril.to(
        device)
    predicted_multivariate_normal_dists._unbroadcasted_scale_tril = predicted_multivariate_normal_dists._unbroadcasted_scale_tril.to(
        device)
    predicted_multivariate_normal_dists.covariance_matrix = predicted_multivariate_normal_dists.covariance_matrix.to(
        device)
    predicted_multivariate_normal_dists.precision_matrix = predicted_multivariate_normal_dists.precision_matrix.to(
        device)

    # Compute negative log probability
    negative_log_prob = - \
        predicted_multivariate_normal_dists.log_prob(gt_box_means)
    negative_log_prob_mean = negative_log_prob.mean()
    output_dict.update({'ignorance_score_mean': negative_log_prob_mean.to(
        device).tolist()})

    # Compute mean square error
    mean_squared_error = ((predicted_box_means - gt_box_means)**2).mean()
    output_dict.update(
        {'mean_squared_error': mean_squared_error.to(device).tolist()})

    # Energy Score.
    sample_set = predicted_multivariate_normal_dists.sample((1001,)).to(device)
    sample_set_1 = sample_set[:-1]
    sample_set_2 = sample_set[1:]

    energy_score = torch.norm(
        (sample_set_1 - gt_box_means),
        dim=2).mean(0) - 0.5 * torch.norm(
        (sample_set_1 - sample_set_2),
        dim=2).mean(0)
    energy_score_mean = energy_score.mean()
    output_dict.update(
        {'energy_score_mean': energy_score_mean.to(device).tolist()})

    return output_dict


def compute_reg_scores_fn(false_negatives, valid_idxs, entropy=True):
    """
    Computes proper scoring rule for regression false positive.

    Args:
        false_negatives (dict): dictionary containing false_negatives
        valid_idxs (tensor): a tensor containing valid element idxs for per-class computation

    Returns:
        output_dict (dict): dictionary containing false positives ignorance and energy scores.
    """
    output_dict = {}

    predicted_box_means = false_negatives['predicted_box_means'][valid_idxs]
    predicted_box_covars = false_negatives['predicted_box_covariances'][valid_idxs]

    predicted_multivariate_normal_dists = torch.distributions.multivariate_normal.MultivariateNormal(
        predicted_box_means.to('cpu'),
        predicted_box_covars.to('cpu') +
        1e-2 * torch.eye(predicted_box_covars.shape[2]).to('cpu'))
    predicted_multivariate_normal_dists.loc = predicted_multivariate_normal_dists.loc.to(
        device)
    predicted_multivariate_normal_dists.scale_tril = predicted_multivariate_normal_dists.scale_tril.to(
        device)
    predicted_multivariate_normal_dists._unbroadcasted_scale_tril = predicted_multivariate_normal_dists._unbroadcasted_scale_tril.to(
        device)
    predicted_multivariate_normal_dists.covariance_matrix = predicted_multivariate_normal_dists.covariance_matrix.to(
        device)
    predicted_multivariate_normal_dists.precision_matrix = predicted_multivariate_normal_dists.precision_matrix.to(
        device)

    # If no valid idxs, do not perform computation
    if predicted_box_means.shape[0] == 0:
        output_dict.update({'total_entropy_mean': None,
                            'total_entropy': None,
                            'fp_energy_score_mean': None})
        return output_dict
    # import ipdb;
    # ipdb.set_trace()
    fp_entropy = predicted_multivariate_normal_dists.entropy()
    fp_entropy_mean = fp_entropy.mean()

    output_dict.update({'total_entropy_mean': fp_entropy_mean.to(
        device).tolist()})
    output_dict.update({'total_entropy': fp_entropy})

    # # Energy Score.
    if entropy:
        sample_set = predicted_multivariate_normal_dists.sample((1001,)).to(device)
        sample_set_1 = sample_set[:-1]
        sample_set_2 = sample_set[1:]
        fp_energy_score = torch.norm((sample_set_1 - sample_set_2), dim=2).mean(0)
        fp_energy_score_mean = fp_energy_score.mean()

        output_dict.update({'fp_energy_score_mean': fp_energy_score_mean.to(
            device).tolist()})
    # output_dict.update({'fp_energy_score_mean': torch.Tensor(int(1)).to(
    #     device).tolist()})

    return output_dict
