#!/usr/bin/env python3

import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
from sklearn.utils import resample


def calculate_3d_iou(vol1, vol2):
    intersection = np.logical_and(vol1, vol2).sum()
    union = np.logical_or(vol1, vol2).sum()
    if union == 0:
        return 0
    return intersection / union


def calculate_3d_hd95_from_points(vol1_points, vol2_points):
    if not vol1_points.size or not vol2_points.size:
        return np.inf

    distances = cdist(vol1_points, vol2_points, metric="euclidean").astype(np.float32)
    d1 = np.percentile(np.min(distances, axis=1), 95)
    d2 = np.percentile(np.min(distances, axis=0), 95)
    return max(d1, d2)


def calculate_3d_assd_from_points(vol1_points, vol2_points):
    if not vol1_points.size or not vol2_points.size:
        return np.inf

    distances = cdist(vol1_points, vol2_points, metric="euclidean").astype(np.float32)
    assd1 = np.mean(np.min(distances, axis=1))
    assd2 = np.mean(np.min(distances, axis=0))
    return (assd1 + assd2) / 2


def match_labels_single_bone(gt_volume, pred_volume, label_range):
    matches = {}

    for label in label_range:
        gt_mask = gt_volume == label
        if gt_mask.any():
            iou_scores = {
                pred_label: calculate_3d_iou(gt_mask, pred_volume == pred_label)
                for pred_label in label_range
                if (pred_volume == pred_label).any()
            }

            if iou_scores:
                best_match = max(iou_scores, key=iou_scores.get)
                matches[label] = (best_match, iou_scores[best_match])

    return matches


def match_labels_whole_pelvis(gt_volume, pred_volume):
    sa_matches = match_labels_single_bone(gt_volume, pred_volume, range(1, 11))
    li_matches = match_labels_single_bone(gt_volume, pred_volume, range(11, 21))
    ri_matches = match_labels_single_bone(gt_volume, pred_volume, range(21, 31))
    return sa_matches | li_matches | ri_matches


def extract_surface_points(volume, label, pixel_spacing, sample_size=10000):
    mask = volume == label

    struct = generate_binary_structure(3, 1)
    eroded = binary_erosion(mask, structure=struct)
    surface_mask = binary_dilation(mask, structure=struct) & ~eroded

    surface_points = np.argwhere(surface_mask)

    if surface_points.shape[0] > sample_size:
        surface_points = resample(surface_points, n_samples=sample_size, random_state=2024)

    adjusted_points = surface_points * pixel_spacing
    return adjusted_points


def calculate_sphere_radius(volume, label):
    points = np.argwhere(volume == label)
    if points.size == 0:
        return np.inf
    center = np.mean(points, axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    return np.max(radii)


def evaluate_fracture_segmentation(matches, gt_volume, pred_volume, spacing):
    results = {}
    for label in matches:
        if matches[label][1] > 0:
            pred_label, _ = matches[label]
            gt_points = extract_surface_points(gt_volume, label, spacing)
            pred_points = extract_surface_points(pred_volume, pred_label, spacing)
            hd95 = calculate_3d_hd95_from_points(gt_points, pred_points)
            assd = calculate_3d_assd_from_points(gt_points, pred_points)
        else:
            radius = calculate_sphere_radius(gt_volume, label)
            hd95 = 2 * radius
            assd = radius

        results[label] = (matches[label][1], hd95, assd)
    return results


def evaluate_anatomical_segmentation(gt_volume, pred_volume, spacing):
    results = {}
    anatomical_ranges = {
        "SA": range(1, 11),
        "LI": range(11, 21),
        "RI": range(21, 31),
    }
    for bone, label_range in anatomical_ranges.items():
        gt_mask = np.isin(gt_volume, label_range)
        pred_mask = np.isin(pred_volume, label_range)
        iou = calculate_3d_iou(gt_mask, pred_mask)
        gt_points = extract_surface_points(gt_mask, 1, spacing)
        pred_points = extract_surface_points(pred_mask, 1, spacing)
        if pred_points.size:
            hd95 = calculate_3d_hd95_from_points(gt_points, pred_points)
            assd = calculate_3d_assd_from_points(gt_points, pred_points)
        else:
            radius = calculate_sphere_radius(gt_mask, 1)
            hd95 = 2 * radius
            assd = radius
        results[bone] = (iou, hd95, assd)

    return results


def evaluate_3d_single_case(gt_volume, pred_volume, spacing, verbose=False):
    if verbose:
        print("Spacing =", spacing)
        print("Size =", gt_volume.shape)

    matches = match_labels_whole_pelvis(gt_volume, pred_volume)
    if verbose:
        print("Matches and IoU scores:", matches)

    fracture_iou, fracture_hd95, fracture_assd = 0, 0, 0
    count = 0
    fracture_results = evaluate_fracture_segmentation(matches, gt_volume, pred_volume, spacing)
    for label, (iou, hd95, assd) in fracture_results.items():
        if verbose:
            print(f"Label {label}: IoU = {iou}, HD95 = {hd95}, ASSD = {assd}")
        fracture_iou += iou
        fracture_hd95 += hd95
        fracture_assd += assd
        count += 1

    fracture_iou = fracture_iou / count
    fracture_hd95 = fracture_hd95 / count
    fracture_assd = fracture_assd / count

    anatomical_iou, anatomical_hd95, anatomical_assd = 0, 0, 0
    anatomical_results = evaluate_anatomical_segmentation(gt_volume, pred_volume, spacing)
    for label, (iou, hd95, assd) in anatomical_results.items():
        if verbose:
            print(f"Label {label}: IoU = {iou}, HD95 = {hd95}, ASSD = {assd}")
        anatomical_iou += iou
        anatomical_hd95 += hd95
        anatomical_assd += assd

    anatomical_iou = anatomical_iou / 3
    anatomical_hd95 = anatomical_hd95 / 3
    anatomical_assd = anatomical_assd / 3

    return {
        "fracture_iou": fracture_iou,
        "fracture_hd95": fracture_hd95,
        "fracture_assd": fracture_assd,
        "anatomical_iou": anatomical_iou,
        "anatomical_hd95": anatomical_hd95,
        "anatomical_assd": anatomical_assd,
    }
