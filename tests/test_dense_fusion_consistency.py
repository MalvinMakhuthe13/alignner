import tempfile
import unittest
from pathlib import Path

from src.reconstruction_engine.config import ReconstructionConfig
from src.reconstruction_engine.openmvs import OpenMVSCommands


class DenseFusionConsistencyTests(unittest.TestCase):
    def test_config_exposes_conservative_multiview_consistency_defaults(self):
        config = ReconstructionConfig()

        self.assertEqual(config.densify_num_views_fuse, 2)
        self.assertEqual(config.densify_geometric_iters, 2)
        self.assertEqual(config.fusion_depth_diff_threshold, 0.01)
        self.assertEqual(config.fusion_reprojection_threshold, 1.2)

    def test_densify_command_passes_consistency_controls(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            executable = Path(temporary_directory) / "DensifyPointCloud.exe"
            executable.touch()

            config = ReconstructionConfig(
                densify_num_views=5,
                densify_num_views_fuse=3,
                densify_geometric_iters=4,
                fusion_depth_diff_threshold=0.015,
                fusion_reprojection_threshold=1.5,
                num_threads=3,
            )

            command = OpenMVSCommands(temporary_directory).densify(
                "scene.mvs",
                config,
            )

        expected = {
            "--number-views": "5",
            "--number-views-fuse": "3",
            "--geometric-iters": "4",
            "--fusion-depth-diff-threshold": "0.015",
            "--fusion-reprojection-threshold": "1.5",
            "--max-threads": "3",
        }

        for option, value in expected.items():
            with self.subTest(option=option):
                index = command.index(option)
                self.assertEqual(command[index + 1], value)

    def test_invalid_fusion_settings_are_rejected(self):
        invalid_settings = (
            {"densify_num_views_fuse": 1},
            {"densify_geometric_iters": -1},
            {"fusion_depth_diff_threshold": 0.0},
            {"fusion_depth_diff_threshold": float("nan")},
            {"fusion_reprojection_threshold": -0.1},
        )

        for settings in invalid_settings:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    ReconstructionConfig(**settings)


if __name__ == "__main__":
    unittest.main()
