from __future__ import annotations

import pickle

from simulator.cli.run import _save_dataset


class PickleFallbackDataset:
    def to_netcdf(self, path) -> None:
        raise RuntimeError("forced NetCDF failure")


def test_save_dataset_falls_back_to_pickle_when_netcdf_fails(tmp_path) -> None:
    dataset = PickleFallbackDataset()

    output_path = _save_dataset(
        dataset,
        output_dir=tmp_path,
        file_stem="example",
    )

    assert output_path == tmp_path / "example.pkl"
    assert output_path.exists()

    with output_path.open("rb") as file:
        restored = pickle.load(file)

    assert isinstance(restored, PickleFallbackDataset)
