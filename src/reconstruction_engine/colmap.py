from pathlib import Path

from src.reconstruction_engine.config import ReconstructionConfig


class COLMAPCommands:
    def __init__(self, executable: str):
        self.executable = str(Path(executable))

    def feature_extractor(
        self,
        database_path: str,
        image_path: str,
        config: ReconstructionConfig,
    ) -> list[str]:
        return [
            self.executable,
            "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--FeatureExtraction.use_gpu", "0",
            "--FeatureExtraction.num_threads", str(config.num_threads),
            "--FeatureExtraction.max_image_size", str(config.max_image_size),
            "--SiftExtraction.max_num_features", str(config.max_num_features),
        ]

    def exhaustive_matcher(
        self,
        database_path: str,
        config: ReconstructionConfig,
    ) -> list[str]:
        return [
            self.executable,
            "exhaustive_matcher",
            "--database_path", str(database_path),
            "--FeatureMatching.use_gpu", "0",
            "--FeatureMatching.num_threads", str(config.num_threads),
            "--FeatureMatching.guided_matching",
            "1" if config.guided_matching else "0",
        ]

    def sequential_matcher(
        self,
        database_path: str,
        config: ReconstructionConfig,
    ) -> list[str]:
        return [
            self.executable,
            "sequential_matcher",
            "--database_path", str(database_path),
            "--FeatureMatching.use_gpu", "0",
            "--FeatureMatching.num_threads", str(config.num_threads),
            "--FeatureMatching.guided_matching",
            "1" if config.guided_matching else "0",
            "--SequentialMatching.overlap",
            str(config.sequential_overlap),
            "--SequentialMatching.loop_detection",
            "1" if config.sequential_loop_detection else "0",
            "--SequentialMatching.num_threads",
            str(config.num_threads),
        ]

    def spatial_matcher(
        self,
        database_path: str,
        config: ReconstructionConfig,
    ) -> list[str]:
        return [
            self.executable,
            "spatial_matcher",
            "--database_path", str(database_path),
            "--FeatureMatching.use_gpu", "0",
            "--FeatureMatching.num_threads", str(config.num_threads),
            "--FeatureMatching.guided_matching",
            "1" if config.guided_matching else "0",
            "--SpatialMatching.ignore_z",
            "1" if config.spatial_ignore_z else "0",
            "--SpatialMatching.max_num_neighbors",
            str(config.spatial_max_num_neighbors),
            "--SpatialMatching.min_num_neighbors",
            str(config.spatial_min_num_neighbors),
            "--SpatialMatching.max_distance",
            str(config.spatial_max_distance),
        ]

    def match(
        self,
        database_path: str,
        config: ReconstructionConfig,
    ) -> list[str]:
        matchers = {
            "exhaustive": self.exhaustive_matcher,
            "sequential": self.sequential_matcher,
            "spatial": self.spatial_matcher,
        }

        try:
            matcher = matchers[config.matcher]
        except KeyError:
            raise ValueError(
                f"Unsupported matcher: {config.matcher}"
            ) from None

        return matcher(database_path, config)

    def mapper(
        self,
        database_path: str,
        image_path: str,
        output_path: str,
        config: ReconstructionConfig,
    ) -> list[str]:
        return [
            self.executable,
            "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--output_path", str(output_path),
            "--Mapper.min_num_matches", str(config.min_num_matches),
            "--Mapper.multiple_models",
            "1" if config.multiple_models else "0",
            "--Mapper.num_threads", str(config.num_threads),
        ]

    def image_undistorter(
        self,
        image_path: str,
        input_path: str,
        output_path: str,
    ) -> list[str]:
        return [
            self.executable,
            "image_undistorter",
            "--image_path", str(image_path),
            "--input_path", str(input_path),
            "--output_path", str(output_path),
            "--output_type", "COLMAP",
        ]
