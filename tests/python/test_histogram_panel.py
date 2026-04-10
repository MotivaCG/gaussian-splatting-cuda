# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for histogram metric extraction."""

from types import SimpleNamespace

import pytest


class _ModelStub:
    def __init__(self, lf, means, scaling, opacity=None):
        self._means = lf.Tensor.from_numpy(means)
        self._scaling = lf.Tensor.from_numpy(scaling)
        self._opacity = None if opacity is None else lf.Tensor.from_numpy(opacity)

    def get_means(self):
        return self._means

    def get_scaling(self):
        return self._scaling

    def get_opacity(self):
        if self._opacity is None:
            raise AssertionError("Opacity should not be requested in this test")
        return self._opacity


def _translation_matrix(tx: float, ty: float, tz: float) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, tx],
        [0.0, 1.0, 0.0, ty],
        [0.0, 0.0, 1.0, tz],
        [0.0, 0.0, 0.0, 1.0],
    ]


@pytest.fixture
def histogram_panel_module():
    from lfs_plugins import histogram_panel

    return histogram_panel


def test_histogram_metrics_include_positions_volume_anisotropy_and_erank(histogram_panel_module):
    metric_ids = {metric.id for metric in histogram_panel_module.METRICS}

    assert {"position_x", "position_y", "position_z", "volume", "anisotropy", "erank"} <= metric_ids


def test_histogram_position_metrics_use_world_space_means(histogram_panel_module, lf, numpy):
    panel = histogram_panel_module.HistogramPanel()
    model = _ModelStub(
        lf,
        numpy.array([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.5]], dtype=numpy.float32),
        numpy.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]], dtype=numpy.float32),
    )

    splat_type = getattr(getattr(lf, "NodeType", None), "SPLAT", None)
    if splat_type is None:
        splat_type = lf.scene.NodeType.SPLAT

    scene = SimpleNamespace(
        get_nodes=lambda: [
            SimpleNamespace(
                id=7,
                parent_id=-1,
                visible=True,
                type=splat_type,
                gaussian_count=2,
                world_transform=_translation_matrix(10.0, -3.0, 0.5),
            )
        ]
    )

    panel._metric_id = "position_x"
    numpy.testing.assert_allclose(
        panel._extract_metric_values(scene, model).cpu().numpy(),
        numpy.array([11.0, 8.0], dtype=numpy.float32),
    )

    panel._metric_id = "position_y"
    numpy.testing.assert_allclose(
        panel._extract_metric_values(scene, model).cpu().numpy(),
        numpy.array([-1.0, -2.5], dtype=numpy.float32),
    )

    panel._metric_id = "position_z"
    numpy.testing.assert_allclose(
        panel._extract_metric_values(scene, model).cpu().numpy(),
        numpy.array([3.5, 5.0], dtype=numpy.float32),
    )


def test_histogram_volume_anisotropy_and_erank_metrics_match_gaussian_scales(histogram_panel_module, lf, numpy):
    panel = histogram_panel_module.HistogramPanel()
    model = _ModelStub(
        lf,
        numpy.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=numpy.float32),
        numpy.array([[1.0, 1.0, 1.0], [1.0, 2.0, 4.0]], dtype=numpy.float32),
    )
    scene = SimpleNamespace(get_nodes=lambda: [])

    panel._metric_id = "volume"
    numpy.testing.assert_allclose(
        panel._extract_metric_values(scene, model).cpu().numpy(),
        numpy.array([4.0 * numpy.pi / 3.0, 32.0 * numpy.pi / 3.0], dtype=numpy.float32),
        rtol=1e-6,
    )

    panel._metric_id = "anisotropy"
    anisotropy = panel._extract_metric_values(scene, model).cpu().numpy()
    numpy.testing.assert_allclose(anisotropy, numpy.array([1.0, 4.0], dtype=numpy.float32), rtol=1e-6)
    assert panel._histogram_bounds(lf.Tensor.from_numpy(anisotropy)) == (1.0, 4.0)

    panel._metric_id = "erank"
    erank = panel._extract_metric_values(scene, model).cpu().numpy()
    numpy.testing.assert_allclose(erank, numpy.array([3.0, 1.9503675], dtype=numpy.float32), rtol=1e-6)
    assert panel._histogram_bounds(lf.Tensor.from_numpy(erank)) == (1.0, 3.0)


def test_compare_heatmap_reuses_primary_metric_and_selects_joint_cells(histogram_panel_module, lf, numpy):
    panel = histogram_panel_module.HistogramPanel()
    panel._metric_id = "volume"
    panel._compare_metric_id = "opacity"

    model = _ModelStub(
        lf,
        numpy.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=numpy.float32),
        numpy.array([[1.0, 1.0, 1.0], [1.5, 1.5, 1.5], [3.0, 3.0, 3.0]], dtype=numpy.float32),
        opacity=numpy.array([0.05, 0.55, 0.95], dtype=numpy.float32),
    )
    scene = SimpleNamespace(get_nodes=lambda: [])

    primary_values = panel._extract_metric_values(scene, model, panel._metric_id)
    panel._refresh_compare(scene, model, primary_values, None)

    assert panel._show_compare_card is True
    assert panel._show_compare_chart is True
    assert panel._compare_x_metric_label == "Volume"
    assert panel._compare_y_metric_label == "Opacity"
    assert panel._compare_counts is not None
    assert sum(panel._compare_counts) == 3
    assert len(panel._compare_counts) == histogram_panel_module.DEFAULT_COMPARE_X_BIN_COUNT ** 2

    x_bins = panel._compare_x_bin_indices.cpu().tolist()
    y_bins = panel._compare_y_bin_indices.cpu().tolist()
    mask = panel._selection_mask_for_compare_bounds(x_bins[0], x_bins[0], y_bins[0], y_bins[0])
    numpy.testing.assert_array_equal(mask.cpu().numpy(), numpy.array([True, False, False]))


def test_histogram_bin_slider_rebins_and_preserves_marked_range(histogram_panel_module, lf, numpy):
    panel = histogram_panel_module.HistogramPanel()
    panel._show_chart = True
    panel._metric_id = "opacity"

    values = lf.Tensor.from_numpy(numpy.array([0.05, 0.15, 0.35, 0.65, 0.85], dtype=numpy.float32))
    finite_mask = values.isfinite()
    panel._primary_values = values
    panel._primary_finite_mask = finite_mask
    panel._primary_valid_values = values[finite_mask]
    panel._primary_histogram_min = 0.0
    panel._primary_histogram_max = 1.0
    panel._histogram_bin_count = 16
    panel._rebuild_histogram_from_cache()

    panel._marked_bin_start = 4
    panel._marked_bin_end = 7
    panel._sync_marked_range(apply_scene=False)

    panel._set_histogram_bin_count(32)

    assert len(panel._hist_counts) == 32
    assert panel._marked_bounds() == (8, 15)
    assert panel._peak_text == "1"


def test_histogram_selection_geometry_accounts_for_bar_gaps(histogram_panel_module):
    panel = histogram_panel_module.HistogramPanel()
    panel._histogram_bin_count = 4
    panel._chart_el = SimpleNamespace(absolute_left=10.0, absolute_width=100.0)

    left, width = panel._histogram_selection_geometry(1, 2)

    assert left == pytest.approx(25.5)
    assert width == pytest.approx(49.0)
    assert panel._bin_index_for_mouse_x(40.0) == 1
    assert panel._bin_index_for_mouse_x(65.0) == 2


def test_compare_bin_sliders_support_rectangular_grids(histogram_panel_module, lf, numpy):
    panel = histogram_panel_module.HistogramPanel()
    panel._metric_id = "volume"
    panel._compare_metric_id = "opacity"

    model = _ModelStub(
        lf,
        numpy.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=numpy.float32),
        numpy.array([[1.0, 1.0, 1.0], [1.5, 1.5, 1.5], [3.0, 3.0, 3.0]], dtype=numpy.float32),
        opacity=numpy.array([0.05, 0.55, 0.95], dtype=numpy.float32),
    )
    scene = SimpleNamespace(get_nodes=lambda: [])

    primary_values = panel._extract_metric_values(scene, model, panel._metric_id)
    panel._refresh_compare(scene, model, primary_values, None)
    panel._set_compare_x_bin_count(12)
    panel._set_compare_y_bin_count(9)

    records = list(panel._build_compare_bin_records())

    assert len(panel._compare_counts) == 12 * 9
    assert len(records) == 12 * 9
    assert len(panel._compare_x_edges) == 13
    assert len(panel._compare_y_edges) == 10
    assert "width: 8.3333%;" in records[0]["style_attr"]
    assert "height: 11.1111%;" in records[0]["style_attr"]
    assert panel._format_compare_bin_count_text() == "12 x 9 bins"
