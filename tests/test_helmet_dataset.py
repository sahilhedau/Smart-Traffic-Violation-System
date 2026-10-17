import pytest
from pathlib import Path
from scripts.train_helmet import validate_split, generate_yaml, find_dataset

def test_helmet_dataset_detection_and_validation():
    try:
        ds_path = find_dataset()
        assert ds_path.exists()
        
        train_stats = validate_split("train", ds_path / "train" / "images", ds_path / "train" / "labels")
        assert train_stats["count"] > 0
        assert train_stats["labels"] > 0
        assert train_stats["class_counts"].get(0, 0) > 0  # 'With Helmet'
        assert train_stats["class_counts"].get(1, 0) > 0  # 'Without Helmet'
    except FileNotFoundError:
        pytest.skip("Helmet dataset not located in local test environment")

def test_helmet_yaml_generation(tmp_path):
    fake_dataset = tmp_path / "fake_dataset"
    (fake_dataset / "train" / "images").mkdir(parents=True)
    (fake_dataset / "valid" / "images").mkdir(parents=True)
    (fake_dataset / "test" / "images").mkdir(parents=True)
    
    out_yaml = tmp_path / "test_helmet.yaml"
    generate_yaml(fake_dataset, out_yaml)
    assert out_yaml.exists()
    
    content = out_yaml.read_text(encoding="utf-8")
    assert "nc: 2" in content
    assert "'With Helmet', 'Without Helmet'" in content
