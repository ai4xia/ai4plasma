import numpy as np
import torch

from visualize_sliding_density_reconstruction import (
    bidirectional_animation_events,
    build_bidirectional_rows,
    project_time_z,
    refine_reconstruction_pass,
    reconstruct_with_slide_step,
    shifted_window_starts,
    sliding_window_starts,
    window_outline_bounds,
)


class RecordingDensityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, model_input):
        self.inputs.append(model_input.detach().clone())
        batch, _channels, time, size_x, size_z = model_input.shape
        prediction = torch.zeros(
            (batch, 4, time, size_x, size_z),
            device=model_input.device,
            dtype=model_input.dtype,
        )
        prediction[:, 3] = 5.0
        return prediction


class IncrementDensityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, model_input):
        self.inputs.append(model_input.detach().clone())
        batch, _channels, time, size_x, size_z = model_input.shape
        prediction = torch.zeros(
            (batch, 4, time, size_x, size_z),
            device=model_input.device,
            dtype=model_input.dtype,
        )
        prediction[:, 3] = model_input[:, 3] + 1.0
        return prediction


def test_forward_schedule_keeps_exact_step_and_crops_padded_tail():
    starts = sliding_window_starts(run_length=150, window_size=24, slide_step=24)
    assert starts == [0, 24, 48, 72, 96, 120, 144]
    assert np.diff(starts).tolist() == [24] * 6
    assert starts[-1] + 24 > 150

    shifted = shifted_window_starts(
        run_length=150,
        window_size=24,
        slide_step=24,
        offset=12,
    )
    assert shifted == [12, 36, 60, 84, 108, 132]
    shifted_with_boundary = shifted_window_starts(
        run_length=150,
        window_size=24,
        slide_step=24,
        offset=12,
        include_boundary_padding=True,
    )
    assert shifted_with_boundary == [-12, 12, 36, 60, 84, 108, 132]

    starts = sliding_window_starts(run_length=150, window_size=24, slide_step=12)
    assert starts[-1] == 132
    assert np.diff(starts).tolist() == [12] * (len(starts) - 1)
    assert starts[-1] + 24 > 150


def test_time_z_projection_averages_x_and_preserves_unreconstructed_nan():
    density = np.asarray(
        [
            [[1.0, 3.0], [3.0, 5.0]],
            [[np.nan, np.nan], [np.nan, np.nan]],
        ]
    )
    averaged = project_time_z(density, x_index=None)
    np.testing.assert_allclose(averaged[0], [2.0, 4.0])
    assert np.isnan(averaged[1]).all()
    np.testing.assert_allclose(
        project_time_z(density, x_index=1)[0],
        [3.0, 5.0],
    )


def test_window_outline_bounds_clip_padding_to_real_run():
    frame_ids = np.arange(6)
    assert window_outline_bounds(
        {"window_start": 2, "valid_end": 6}, frame_ids
    ) == (1.5, 5.5)
    assert window_outline_bounds(
        {"window_start": -2, "valid_start": 0, "valid_end": 2}, frame_ids
    ) == (-0.5, 1.5)
    assert window_outline_bounds(
        {"window_start": 6, "valid_start": 6, "valid_end": 6}, frame_ids
    ) is None
    assert window_outline_bounds(None, frame_ids) is None


def test_probes_are_clamped_only_for_recursive_input_not_output_metrics():
    model = RecordingDensityModel()
    target = torch.zeros((4, 6, 2, 2), dtype=torch.float32)
    target[3] = 2.0
    density_mean = torch.tensor(10.0)
    density_std = torch.tensor(2.0)
    target_density = (target[3] * density_std + density_mean).numpy()
    probe_mask = torch.zeros((2, 2), dtype=torch.float32)
    probe_mask[0, 0] = 1.0

    result = reconstruct_with_slide_step(
        model=model,
        target_normalized=target,
        target_density_plot=target_density,
        density_mean=density_mean,
        density_std=density_std,
        probe_mask=probe_mask,
        window_size=4,
        slide_step=2,
        plot_units="physical",
        x_index=None,
        amp=False,
    )

    assert result["window_starts"] == [0, 2]
    second_input = model.inputs[1]
    density_values = second_input[0, 3]
    density_masks = second_input[0, 7]

    # Overlap uses a complete previous reconstruction, but its true probe is
    # restored before recursion. The new tail exposes only the true probe.
    assert torch.all(density_masks[:2] == 1.0)
    assert torch.all(density_values[:2, 0, 0] == 2.0)
    assert torch.all(density_values[:2, 1, 1] == 5.0)
    assert torch.all(density_masks[2:, 0, 0] == 1.0)
    assert torch.all(density_masks[2:, 1, 1] == 0.0)
    assert torch.all(density_values[2:, 0, 0] == 2.0)

    # Raw normalized probe predictions remain 5 rather than being overwritten
    # by target=2. In physical plot units these values are 20 and 14, while
    # NRMSE remains the direct standardized-space RMS of 5 - 2.
    assert np.all(result["final_reconstruction"] == 20.0)
    assert np.isclose(result["metrics"]["full_run_rmse"], 6.0)
    assert np.isclose(result["metrics"]["full_run_nrmse"], 3.0)
    assert result["metrics"]["final_window_padded_slots"] == 0


def test_reverse_refinement_uses_full_latest_state_and_keeps_raw_probes():
    model = IncrementDensityModel()
    target = torch.zeros((4, 6, 2, 2), dtype=torch.float32)
    target[3] = 2.0
    probe_mask = torch.zeros((2, 2), dtype=torch.float32)
    probe_mask[0, 0] = 1.0
    raw_state = torch.full((6, 2, 2), 5.0)
    conditioning_state = raw_state.clone()
    conditioning_state[:, 0, 0] = 2.0

    pass_info = refine_reconstruction_pass(
        model=model,
        target_normalized=target,
        raw_state_normalized=raw_state,
        conditioning_state_normalized=conditioning_state,
        probe_mask=probe_mask,
        window_size=4,
        slide_step=2,
        direction="reverse",
        offset=0,
        amp=False,
    )

    assert pass_info["ordered_window_starts"] == [2, 0]
    assert torch.all(model.inputs[0][0, 7] == 1.0)
    assert torch.all(model.inputs[0][0, 3, :, 0, 0] == 2.0)
    assert torch.all(model.inputs[0][0, 3, :, 1, 1] == 5.0)

    # The second reverse window immediately sees the first window's newest
    # conditioning output in their overlap.
    second_density = model.inputs[1][0, 3]
    assert torch.all(second_density[2:, 0, 0] == 2.0)
    assert torch.all(second_density[2:, 1, 1] == 6.0)

    # Probe values are true only in conditioning. Raw outputs retain model=3.
    assert torch.all(raw_state[:, 0, 0] == 3.0)
    assert torch.all(conditioning_state[:, 0, 0] == 2.0)


def test_bidirectional_rows_match_requested_six_experiments():
    model = IncrementDensityModel()
    target = torch.zeros((4, 6, 2, 2), dtype=torch.float32)
    target[3] = 2.0
    probe_mask = torch.zeros((2, 2), dtype=torch.float32)
    probe_mask[0, 0] = 1.0

    rows = build_bidirectional_rows(
        model=model,
        target_normalized=target,
        target_density_plot=target[3].numpy(),
        density_mean=torch.tensor(0.0),
        density_std=torch.tensor(1.0),
        probe_mask=probe_mask,
        window_size=4,
        refinement_step=2,
        refinement_passes=4,
        refinement_offset=2,
        plot_units="normalized",
        x_index=None,
        amp=False,
    )

    assert len(rows) == 6
    assert [row["pass_count"] for row in rows] == [1, 2, 3, 4, 4, 4]
    assert [row["slide_step"] for row in rows] == [2, 2, 2, 2, 4, 4]
    assert rows[4]["offsets"] == [0, 0, 0, 0]
    assert rows[5]["offsets"] == [0, 2, 0, 2]
    assert rows[3]["directions"] == [
        "forward",
        "reverse",
        "forward",
        "reverse",
    ]
    assert [len(row["animation_snapshots"]) for row in rows] == [2, 4, 6, 8, 8, 8]
    assert rows[5]["history"][1]["ordered_window_starts"] == [2, -2]

    events = bidirectional_animation_events(rows)
    assert len(events) == 8
    assert all(
        event["phase"] == "window-call synchronized reconstruction"
        for event in events
    )
    assert all(snapshot is not None for snapshot in events[0]["snapshots"])
    assert "window_start" not in events[2]["snapshots"][0]
    assert "window_start" in events[2]["snapshots"][1]
    assert "synchronized window call=8/8" in events[-1]["label"]
