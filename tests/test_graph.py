from __future__ import annotations

from ct_mcot.modules.graph import grid_laplacian


def test_grid_laplacian_shape() -> None:
    lap = grid_laplacian(2, 3)
    assert lap.shape == (6, 6)
