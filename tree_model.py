
from pathlib import Path
import numpy as np
from PIL import Image

"""Functions extracted unchanged from MMA3001_Assignment.ipynb.

This module contains the original notebook implementations of filename matching,
ground-truth tree geometry, branch statistics, and segmentation evaluation.
The images are represented in pixels or relative image coordinates; no
physical calibration is implied. Extracted functions retain notebook behavior.
"""

def get_observation_id(path):
    """
    Extract a common observation identifier by removing
    the RGB, D or GT label from the filename.
    """

    name = path.stem

    for label in ["_RGB-", "_D-", "_GT-"]:
        if label in name:
            return name.replace(label, "_")

    return None


def extract_tree_geometry(gt_path):
    """
    Extract basic 2D geometric features from a ground-truth
    tree mask.

    Parameters
    ----------
    gt_path : str or Path
        File path to a ground-truth mask.

    Returns
    -------
    dict
        Dictionary containing the extracted geometry features.
    """

    # Load the ground-truth mask as grayscale
    gt_image = Image.open(gt_path).convert("L")

    # Convert image into a NumPy array
    gt_array = np.asarray(gt_image)

    # Convert to binary mask
    # White pixels represent the annotated tree structure
    tree_mask = gt_array > 127

    # Identify coordinates of all tree pixels
    y_coords, x_coords = np.where(tree_mask)

    # Handle empty masks
    if len(x_coords) == 0:
        return None

    # Calculate the bounding box of the annotated structure
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)

    y_min = np.min(y_coords)
    y_max = np.max(y_coords)

    # Calculate width and height in pixels
    tree_width = x_max - x_min + 1
    tree_height = y_max - y_min + 1

    # Calculate width-to-height ratio
    aspect_ratio = tree_width / tree_height

    # Calculate the total annotated area
    tree_area = np.sum(tree_mask)

    # Normalise measurements by image dimensions
    image_height, image_width = tree_mask.shape

    relative_width = tree_width / image_width
    relative_height = tree_height / image_height

    # Return extracted features
    return {
        "tree_width_px": tree_width,
        "tree_height_px": tree_height,
        "aspect_ratio": aspect_ratio,
        "annotated_area_px": tree_area,
        "relative_width": relative_width,
        "relative_height": relative_height
    }


def extract_branch_features(gt_path):
    """
    Estimate branching geometry from a binary GT mask.

    The branching point is approximated by finding where
    the annotated structure becomes wider above the trunk.

    These are image-based measurements, not physical distances.
    """

    # Load the ground-truth mask
    gt = np.asarray(
        Image.open(gt_path).convert("L")
    ) > 127

    height, width = gt.shape

    # Measure the horizontal span of annotated pixels
    # at each row of the image
    row_spans = np.zeros(height)

    for y in range(height):

        x_positions = np.where(gt[y])[0]

        if len(x_positions) > 0:
            row_spans[y] = (
                x_positions.max() - x_positions.min() + 1
            )

    # Estimate typical trunk width using the bottom 15%
    # of the image
    bottom_start = int(height * 0.85)

    trunk_rows = row_spans[bottom_start:]
    trunk_rows = trunk_rows[trunk_rows > 0]

    if len(trunk_rows) == 0:
        return None

    trunk_width = np.median(trunk_rows)

    # Search upwards from the lower trunk.
    # The threshold is an approximate indicator of branching.
    threshold = max(1.8 * trunk_width, trunk_width + 5)

    branch_y = np.nan

    for y in range(height - 1, -1, -1):

        if row_spans[y] > threshold:
            branch_y = y
            break

    # Estimate horizontal branch spread
    y_coords, x_coords = np.where(gt)

    if len(x_coords) == 0:
        return None

    # Use the median trunk position near the bottom
    trunk_pixels_y, trunk_pixels_x = np.where(
        gt[bottom_start:]
    )

    if len(trunk_pixels_x) == 0:
        return None

    trunk_centre = np.median(trunk_pixels_x)

    # Calculate maximum spread on either side
    left_spread = max(
        0,
        trunk_centre - np.min(x_coords)
    )

    right_spread = max(
        0,
        np.max(x_coords) - trunk_centre
    )

    # Calculate asymmetry
    total_spread = left_spread + right_spread

    asymmetry = (
        abs(left_spread - right_spread) / total_spread
        if total_spread > 0 else 0
    )

    return {
        "trunk_width_px": trunk_width,
        "branch_point_y_px": branch_y,
        "branch_point_relative": (
            branch_y / height
            if not np.isnan(branch_y) else np.nan
        ),
        "left_spread_px": left_spread,
        "right_spread_px": right_spread,
        "branch_asymmetry": asymmetry
    }


def curve_statistics(curves):
    """
    Calculate equal-weight summary statistics, ignoring
    unobserved points rather than replacing them with zero.
    """

    count = np.sum(np.isfinite(curves), axis=0)

    mean = np.full(curves.shape[1], np.nan)
    std = np.full(curves.shape[1], np.nan)

    for j in range(curves.shape[1]):

        values = curves[:, j]
        values = values[np.isfinite(values)]

        if len(values) > 0:
            mean[j] = values.mean()
            std[j] = values.std()

    return mean, std, count


def calculate_segmentation_metrics(true_mask, predicted_mask):
    """
    Calculate Dice and IoU between two binary masks.
    """

    true_mask = np.asarray(true_mask, dtype=bool)
    predicted_mask = np.asarray(predicted_mask, dtype=bool)

    intersection = np.logical_and(
        true_mask, predicted_mask
    ).sum()

    total_pixels = true_mask.sum() + predicted_mask.sum()

    union = np.logical_or(
        true_mask, predicted_mask
    ).sum()

    dice = (
        2 * intersection / total_pixels
        if total_pixels > 0 else 1.0
    )

    iou = (
        intersection / union
        if union > 0 else 1.0
    )

    return float(dice), float(iou)


def evaluate_probabilities(probabilities, masks, threshold):
    """
    Calculate mean binary Dice and IoU across a dataset.
    Each image contributes equally.
    """

    dice_scores = []
    iou_scores = []

    for probability, ground_truth in zip(probabilities, masks):

        predicted = probability[..., 0] >= threshold
        actual = ground_truth[..., 0] > 0.5

        dice, iou = calculate_segmentation_metrics(
            actual, predicted
        )

        dice_scores.append(dice)
        iou_scores.append(iou)

    return np.mean(dice_scores), np.mean(iou_scores)

