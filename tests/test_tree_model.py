
"""Unit tests for unchanged functions extracted from the MMA3001 notebook."""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from tree_model import (get_observation_id, extract_tree_geometry,
                        extract_branch_features, curve_statistics,
                        calculate_segmentation_metrics, evaluate_probabilities)

@pytest.fixture
def gt_mask_path(tmp_path):
    """A synthetic 20x20 tree mask with a 2-pixel trunk and fork."""
    m=np.zeros((20,20),dtype=np.uint8)
    m[10:20,9:11]=255
    m[5:10,5:7]=255
    m[5:10,13:15]=255
    p=tmp_path/'test_GT-1.png'
    Image.fromarray(m).save(p)
    return p

def test_observation_id_rgb():
    assert get_observation_id(Path('R01N01_Summer_RGB-12-40-50.png'))=='R01N01_Summer_12-40-50'

def test_observation_id_depth_and_gt():
    assert get_observation_id(Path('R01N01_Summer_D-12-40-50.png'))=='R01N01_Summer_12-40-50'
    assert get_observation_id(Path('R01N01_Summer_GT-12-40-50.png'))=='R01N01_Summer_12-40-50'

def test_observation_id_unknown():
    assert get_observation_id(Path('unrecognised.png')) is None

def test_geometry_known_mask(gt_mask_path):
    g=extract_tree_geometry(gt_mask_path)
    assert g['tree_width_px']==10
    assert g['tree_height_px']==15
    assert g['annotated_area_px']==40
    assert g['relative_width']==pytest.approx(0.5)
    assert g['relative_height']==pytest.approx(0.75)
    assert g['aspect_ratio']==pytest.approx(10/15)

def test_geometry_empty_returns_none(tmp_path):
    p=tmp_path/'empty.png'
    Image.fromarray(np.zeros((8,8),dtype=np.uint8)).save(p)
    assert extract_tree_geometry(p) is None

def test_branch_features_known_mask(gt_mask_path):
    g=extract_branch_features(gt_mask_path)
    assert g is not None
    assert g['trunk_width_px']==pytest.approx(2)
    assert g['branch_point_y_px']==9
    assert g['left_spread_px']==pytest.approx(4.5)
    assert g['right_spread_px']==pytest.approx(4.5)
    assert g['branch_asymmetry']==pytest.approx(0)

def test_branch_features_empty_returns_none(tmp_path):
    p=tmp_path/'empty.png'
    Image.fromarray(np.zeros((8,8),dtype=np.uint8)).save(p)
    assert extract_branch_features(p) is None

def test_curve_statistics_ignores_nan():
    values=np.array([[1.0,np.nan,4.0],[3.0,2.0,np.nan]])
    mean,std,count=curve_statistics(values)
    np.testing.assert_allclose(mean,[2,2,4])
    np.testing.assert_allclose(std,[1,0,0])
    np.testing.assert_array_equal(count,[2,1,1])

def test_curve_statistics_all_missing_column():
    values=np.array([[np.nan,2.],[np.nan,4.]])
    mean,std,count=curve_statistics(values)
    assert np.isnan(mean[0]) and np.isnan(std[0])
    assert count[0]==0
    assert mean[1]==pytest.approx(3)

def test_metrics_identical():
    m=np.array([[1,0],[1,1]])
    assert calculate_segmentation_metrics(m,m)==(1.,1.)

def test_metrics_disjoint():
    a=np.array([[1,0],[0,0]])
    b=np.array([[0,0],[0,1]])
    assert calculate_segmentation_metrics(a,b)==(0.,0.)

def test_metrics_partial_overlap():
    a=np.array([[1,1],[0,0]])
    b=np.array([[1,0],[1,0]])
    dice,iou=calculate_segmentation_metrics(a,b)
    assert dice==pytest.approx(0.5)
    assert iou==pytest.approx(1/3)

def test_metrics_both_empty():
    e=np.zeros((3,3))
    assert calculate_segmentation_metrics(e,e)==(1.,1.)

def test_evaluate_probabilities_with_known_threshold():
    p=np.array([[[[0.9],[0.2]],[[0.7],[0.1]]]],dtype=float)
    gt=np.array([[[[1],[0]],[[1],[0]]]],dtype=float)
    dice,iou=evaluate_probabilities(p,gt,0.5)
    assert dice==pytest.approx(1.)
    assert iou==pytest.approx(1.)

def test_evaluate_probabilities_high_threshold():
    p=np.array([[[[0.9],[0.2]],[[0.7],[0.1]]]],dtype=float)
    gt=np.array([[[[1],[0]],[[1],[0]]]],dtype=float)
    dice,iou=evaluate_probabilities(p,gt,0.8)
    assert dice==pytest.approx(2/3)
    assert iou==pytest.approx(1/2)
