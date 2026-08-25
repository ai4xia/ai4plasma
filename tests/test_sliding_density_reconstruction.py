import numpy as np
import torch

from visualize_sliding_density_reconstruction import (
    project_time_z,
    reconstruct_with_slide_step,
    sliding_window_starts,
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


def test_forward_schedule_keeps_exact_step_and_crops_padded_tail():
    starts = sliding_window_starts(run_length=150, window_size=24, slide_step=24)
    assert starts == [0, 24, 48, 72, 96, 120, 144]
    assert np.diff(starts).tolist() == [24] * 6
    assert starts[-1] + 24 > 150

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


def test_probes_are_clamped_only_for_recursive_input_not_output_metrics():
    model = RecordingDensityModel()
    target = torch.zeros((4, 6, 2, 2), dtype=torch.float32)
    target[3] = 2.0
    target_density = target[3].numpy()
    probe_mask = torch.zeros((2, 2), dtype=torch.float32)
    probe_mask[0, 0] = 1.0

    result = reconstruct_with_slide_step(
        model=model,
        target_normalized=target,
        target_density_plot=target_density,
        density_mean=torch.tensor(0.0),
        density_std=torch.tensor(1.0),
        probe_mask=probe_mask,
        window_size=4,
        slide_step=2,
        plot_units="normalized",
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

    # Raw probe predictions remain 5 rather than being overwritten by target=2.
    assert np.all(result["final_reconstruction"] == 5.0)
    assert np.isclose(result["metrics"]["full_run_rmse"], 3.0)
    assert result["metrics"]["final_window_padded_slots"] == 0
