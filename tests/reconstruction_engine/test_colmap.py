import pytest

from src.reconstruction_engine.config import ReconstructionConfig
from src.reconstruction_engine.colmap import COLMAPCommands


@pytest.fixture
def commands():
    return COLMAPCommands(r"D:\colmap\bin\colmap.exe")


@pytest.mark.parametrize(
    "matcher,expected_command",
    [
        ("exhaustive", "exhaustive_matcher"),
        ("sequential", "sequential_matcher"),
        ("spatial", "spatial_matcher"),
    ],
)
def test_matcher_dispatch(commands, matcher, expected_command):
    config = ReconstructionConfig(matcher=matcher)

    result = commands.match("database.db", config)

    assert result[1] == expected_command
    assert "--database_path" in result
    assert "database.db" in result


def test_sequential_overlap_is_configurable(commands):
    config = ReconstructionConfig(
        matcher="sequential",
        sequential_overlap=15,
        sequential_loop_detection=True,
    )

    result = commands.match("database.db", config)

    assert result[result.index("--SequentialMatching.overlap") + 1] == "15"
    assert result[result.index("--SequentialMatching.loop_detection") + 1] == "1"


def test_spatial_parameters_are_configurable(commands):
    config = ReconstructionConfig(
        matcher="spatial",
        spatial_max_num_neighbors=30,
        spatial_min_num_neighbors=5,
        spatial_max_distance=75.0,
    )

    result = commands.match("database.db", config)

    assert result[result.index("--SpatialMatching.max_num_neighbors") + 1] == "30"
    assert result[result.index("--SpatialMatching.min_num_neighbors") + 1] == "5"
    assert result[result.index("--SpatialMatching.max_distance") + 1] == "75.0"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"matcher": "invalid"},
        {"sequential_overlap": 0},
        {"spatial_max_num_neighbors": 0},
        {"spatial_min_num_neighbors": 10, "spatial_max_num_neighbors": 5},
        {"spatial_max_distance": 0},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ReconstructionConfig(**kwargs)
