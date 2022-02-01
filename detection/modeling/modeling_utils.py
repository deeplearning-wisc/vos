import torch


def covariance_output_to_cholesky(pred_bbox_cov):
    """
    Transforms output to covariance cholesky decomposition.
    Args:
        pred_bbox_cov (kx4 or kx10): Output covariance matrix elements.

    Returns:
        predicted_cov_cholesky (kx4x4): cholesky factor matrix
    """
    # Embed diagonal variance
    diag_vars = torch.sqrt(torch.exp(pred_bbox_cov[:, 0:4]))
    predicted_cov_cholesky = torch.diag_embed(diag_vars)
    # import ipdb;
    # ipdb.set_trace()
    if pred_bbox_cov.shape[1] > 4:
        # print('hhh')
        tril_indices = torch.tril_indices(row=4, col=4, offset=-1)
        predicted_cov_cholesky[:, tril_indices[0],
                               tril_indices[1]] = pred_bbox_cov[:, 4:]

    return predicted_cov_cholesky


def clamp_log_variance(pred_bbox_cov, clamp_min=-7.0, clamp_max=7.0):
    """
    Tiny function that clamps variance for consistency across all methods.
    """
    pred_bbox_var_component = torch.clamp(
        pred_bbox_cov[:, 0:4], clamp_min, clamp_max)
    return torch.cat((pred_bbox_var_component, pred_bbox_cov[:, 4:]), dim=1)


def get_probabilistic_loss_weight(current_step, annealing_step):
    """
    Tiny function to get adaptive probabilistic loss weight for consistency across all methods.
    """
    probabilistic_loss_weight = min(1.0, current_step / annealing_step)
    probabilistic_loss_weight = (
        100 ** probabilistic_loss_weight - 1.0) / (100.0 - 1.0)

    return probabilistic_loss_weight
