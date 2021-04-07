import torch
import numpy as np


def l2_distance(feature1, feature2):
    diff = feature1 - feature2
    diff = (diff * diff).mean(dim=1).mean(dim=1).mean(dim=1)
    return diff


def calc_l2_dist_torch(feature1, feature2, dim=1, is_sqrt=False, is_neg=False):
    XX = torch.sum(feature1*feature1, dim=dim, keepdim=True)
    if dim == 1:
        XY = torch.mm(feature1, feature2.T)
        YY = torch.sum(feature2*feature2, dim=dim, keepdim=True).T
    else:
        XY = torch.bmm(feature1, feature2.permute(0, 2, 1))
        YY = torch.sum(feature2*feature2, dim=dim,
                       keepdim=True).permute(0, 2, 1)
    # print(XX.shape, XY.shape, YY.shape, feature1.shape)

    dist = XX - 2 * XY + YY
    # dist = torch.sqrt(dist.abs())
    dist = dist.clamp(min=0)
    if is_sqrt:
        dist = torch.sqrt(dist)

    if is_neg:
        dist = - dist

    return dist


def calc_l2_mean_dist_torch(feature1, feature2, reduce="max", dim=2):
    if dim == 2:
        if reduce == "max":
            feature1 = feature1.max(dim=2)[0].max(dim=2)[0]
            feature2 = feature2.max(dim=2)[0].max(dim=2)[0]
        elif reduce == "mean":
            feature1 = feature1.mean(dim=2).mean(dim=2)
            feature2 = feature2.mean(dim=2).mean(dim=2)
    elif dim == 1:
        B = feature1.shape[0]
        feature1 = feature1.reshape(B, -1)
        B = feature1.shape[0]
        feature2 = feature2.reshape(B, -1)
        if reduce == "max":
            feature1 = feature1.max(dim=1, keepdim=True)[0]
            feature2 = feature2.max(dim=1, keepdim=True)[0]
        elif reduce == "mean":
            feature1 = feature1.mean(dim=1, keepdim=True)
            feature2 = feature2.mean(dim=1, keepdim=True)

    diff = calc_l2_dist_torch(feature1, feature2, dim=1)
    return diff


def get_reduced_tensor(feature1, reduce="max"):
    if reduce == "max":
        feature1 = feature1.max(dim=2)[0].max(dim=2)[0]
    elif reduce == "mean":
        feature1 = feature1.mean(dim=2).mean(dim=2)

    return feature1


def pearson_corr_torch(feat1, feat2):
    B, D = feat1.shape
    mean1 = feat1.mean(dim=1, keepdim=True)
    diff1 = feat1 - mean1
    std1 = diff1.std(dim=1, keepdim=True, unbiased=False)
    diff1 = diff1 / std1

    mean2 = feat2.mean(dim=1, keepdim=True)
    diff2 = feat2 - mean2
    std2 = diff2.std(dim=1, keepdim=True, unbiased=False)
    diff2 = diff2 / std2

#     print(diff2.shape, std2.shape)
    cov = torch.mm(diff1, diff2.permute(1, 0)) / D
#     normalize = torch.mm(std1, std2.permute(1, 0))

#     coef = cov / normalize

    return cov


def calc_gaussian_prob_rate(x, mean_v, sigma_v, label):
    dist_mat = calc_mahalanobis_torch(x, mean_v, sigma_v)
    gt_logit = dist_mat[torch.arange(len(label)), label][:, None]
    logit_mat = gt_logit - dist_mat
    logit_mat = logit_mat.exp()
    ratio = sigma_v[[0]] / sigma_v
    ratio = ratio.sqrt().prod(dim=1)[None, :]
    logit_mat = ratio * logit_mat
    logit_mat = logit_mat.sum(dim=1)
    loss = logit_mat.log().mean()
    return loss
    


def calc_mahalanobis_numpy(feature1, feature2, sigma):
    XX = feature1 * feature1
    XXs = XX[:, None, :] / sigma[None, :, :]
    XXs = np.sum(XXs, axis=2)
    XY = np.dot(feature1, (feature2 / sigma).T)
    YY = (feature2 * feature2 / sigma).sum(axis=1)
    dist = XXs - 2*XY + YY
    return dist


def calc_mahalanobis_torch(feature1, feature2, sigma):
    XX = feature1 * feature1
    XXs = XX[:, None, :] / sigma[None, :, :]
    XXs = torch.sum(XXs, dim=2)
    XY = torch.mm(feature1, (feature2 / sigma).T)
    YY = (feature2 * feature2 / sigma).sum(dim=1)
    dist = XXs - 2*XY + YY
    return dist


def calc_mahalanobis_naive(features1, features2, sigma):
    dist_mat = np.zeros((len(features1), len(features2)))
    for idx, feat1 in enumerate(features1):
        for idx2, feat2 in enumerate(features2):
            diff = feat1 - feat2
            dist = (diff * diff / sigma[idx2]).sum()
            dist_mat[idx, idx2] = dist
    return dist_mat


def calc_cossim_dist_numpy(feature1, feature2):
    feature1 = feature1 / \
        np.sqrt(np.sum(feature1 * feature1, axis=1, keepdims=True))
    feature2 = feature2 / \
        np.sqrt(np.sum(feature2 * feature2, axis=1, keepdims=True))
    distance_mat = np.dot(feature1, feature2.T)
    return distance_mat


def calc_l2_dist_numpy(feature1, feature2, is_sqrt=True, is_neg=False):
    # (Qn, D), (Sn, D)
    feature1 = feature1.astype(float)
    feature2 = feature2.astype(float)
    XX = np.sum(feature1*feature1, axis=1, keepdims=True)
    XY = np.dot(feature1, feature2.T)
    YY = np.sum(feature2*feature2, axis=1, keepdims=True).T
    # print(XX.shape, XY.shape, YY.shape, feature1.shape)

    dist = XX - 2 * XY + YY
    dist = dist.clip(min=0)
    if is_sqrt:
        dist = np.sqrt(dist)

    if is_neg:
        dist = -dist

    return dist
