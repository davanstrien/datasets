import lance
import numpy as np
import pyarrow as pa
import pytest

from datasets import DownloadConfig, load_dataset
from datasets.packaged_modules.lance.lance import _resolve_storage_options


@pytest.fixture
def lance_dataset(tmp_path) -> str:
    data = pa.table(
        {
            "id": pa.array([1, 2, 3, 4]),
            "value": pa.array([10.0, 20.0, 30.0, 40.0]),
            "text": pa.array(["a", "b", "c", "d"]),
            "vector": pa.FixedSizeListArray.from_arrays(pa.array([0.1] * 16, pa.float32()), list_size=4),
        }
    )
    dataset_path = tmp_path / "test_dataset.lance"
    lance.write_dataset(data, dataset_path)
    return str(dataset_path)


@pytest.fixture
def lance_hf_dataset(tmp_path) -> str:
    data = pa.table(
        {
            "id": pa.array([1, 2, 3, 4]),
            "value": pa.array([10.0, 20.0, 30.0, 40.0]),
            "text": pa.array(["a", "b", "c", "d"]),
            "vector": pa.FixedSizeListArray.from_arrays(pa.array([0.1] * 16, pa.float32()), list_size=4),
        }
    )
    dataset_dir = tmp_path / "data" / "train.lance"
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    lance.write_dataset(data, dataset_dir)
    lance.write_dataset(data[:2], tmp_path / "data" / "test.lance")

    with open(tmp_path / "README.md", "w") as f:
        f.write("""---
size_categories:
- 1M<n<10M
source_datasets:
- lance_test
---
    # Test Lance Dataset\n\n
    # My Markdown is fancier\n
""")

    return str(tmp_path)


def test_load_lance_dataset(lance_dataset):
    dataset_dict = load_dataset(lance_dataset)
    assert "train" in dataset_dict.keys()

    dataset = dataset_dict["train"]
    assert "id" in dataset.column_names
    assert "value" in dataset.column_names
    assert "text" in dataset.column_names
    assert "vector" in dataset.column_names
    ids = dataset["id"]
    assert ids == [1, 2, 3, 4]


@pytest.mark.parametrize("streaming", [False, True])
def test_load_hf_dataset(lance_hf_dataset, streaming):
    dataset_dict = load_dataset(lance_hf_dataset, columns=["id", "text"], streaming=streaming)
    assert "train" in dataset_dict.keys()
    assert "test" in dataset_dict.keys()
    dataset = dataset_dict["train"]

    assert "id" in dataset.column_names
    assert "text" in dataset.column_names
    assert "value" not in dataset.column_names
    assert "vector" not in dataset.column_names
    ids = list(dataset["id"])
    assert ids == [1, 2, 3, 4]
    text = list(dataset["text"])
    assert text == ["a", "b", "c", "d"]
    assert "value" not in dataset.column_names


def test_load_vectors(lance_hf_dataset):
    dataset_dict = load_dataset(lance_hf_dataset, columns=["vector"])
    assert "train" in dataset_dict.keys()
    dataset = dataset_dict["train"]

    assert "vector" in dataset.column_names
    vectors = dataset.data["vector"].combine_chunks().values.to_numpy(zero_copy_only=False)
    assert np.allclose(vectors, np.full(16, 0.1))


HF_FILES = ["hf://datasets/user/repo/data/train.lance/_versions/1.manifest"]


def test_resolve_storage_options_local_files():
    assert _resolve_storage_options(["/local/path/train.lance/_versions/1.manifest"], DownloadConfig()) is None


def test_resolve_storage_options_hf_token_from_download_config(monkeypatch):
    # `get_dataset_split_names(..., token=...)` (e.g. the dataset-viewer): storage_options
    # is empty and the token only lives on download_config.token
    monkeypatch.setattr("huggingface_hub.utils._headers.get_token", lambda: None)
    storage_options = _resolve_storage_options(HF_FILES, DownloadConfig(token="hf_secret"))
    assert storage_options == {"token": "hf_secret"}


def test_resolve_storage_options_hf_ambient_token(monkeypatch):
    # No explicit token anywhere: fall back to the locally saved token, like HfFileSystem does
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_DISABLE_IMPLICIT_TOKEN", False)
    monkeypatch.setattr("huggingface_hub.utils._headers.get_token", lambda: "hf_ambient")
    storage_options = _resolve_storage_options(HF_FILES, DownloadConfig())
    assert storage_options == {"token": "hf_ambient"}


def test_resolve_storage_options_drops_none_values(monkeypatch):
    # `load_dataset(..., streaming=True)` populates storage_options["hf"] with token=None,
    # which lance's bindings reject with a TypeError if passed through
    monkeypatch.setattr("huggingface_hub.utils._headers.get_token", lambda: None)
    download_config = DownloadConfig()
    download_config.storage_options = {"hf": {"endpoint": "https://huggingface.co", "token": None}}
    storage_options = _resolve_storage_options(HF_FILES, download_config)
    assert storage_options == {"endpoint": "https://huggingface.co"}


def test_resolve_storage_options_config_token_precedence(monkeypatch):
    # An explicit LanceConfig(token=...) wins over download_config and the ambient token
    monkeypatch.setattr("huggingface_hub.utils._headers.get_token", lambda: "hf_ambient")
    storage_options = _resolve_storage_options(HF_FILES, DownloadConfig(token="hf_dc"), token="hf_config")
    assert storage_options == {"token": "hf_config"}


def test_resolve_storage_options_hf_storage_options_token_overrides_download_config_token():
    # Protocol-specific storage options override the top-level token, like in the fsspec path
    # preparation ({"token": token, **storage_options})
    download_config = DownloadConfig(token="hf_top")
    download_config.storage_options = {"hf": {"token": "hf_storage"}}
    assert _resolve_storage_options(HF_FILES, download_config) == {"token": "hf_storage"}


def test_resolve_storage_options_hf_storage_options_false_disables_auth():
    # storage_options={"hf": {"token": False}} explicitly disables auth even with a top-level token
    download_config = DownloadConfig(token="hf_top")
    download_config.storage_options = {"hf": {"token": False}}
    assert _resolve_storage_options(HF_FILES, download_config) is None


def test_resolve_storage_options_token_true_uses_ambient_token(monkeypatch):
    # token=True means "use the ambient token" (huggingface_hub convention); passing the
    # raw True to lance would raise the same dict[str, str] TypeError
    monkeypatch.setattr("huggingface_hub.utils._headers.get_token", lambda: "hf_ambient")
    storage_options = _resolve_storage_options(HF_FILES, DownloadConfig(token=True))
    assert storage_options == {"token": "hf_ambient"}


def test_resolve_storage_options_token_false_stays_anonymous(monkeypatch):
    # token=False is an explicit opt-out of authentication: don't fall back to the ambient token
    monkeypatch.setattr("huggingface_hub.utils._headers.get_token", lambda: "hf_ambient")
    assert _resolve_storage_options(HF_FILES, DownloadConfig(token=False)) is None


def test_resolve_storage_options_non_hf_protocol_passthrough():
    download_config = DownloadConfig()
    download_config.storage_options = {"s3": {"key": "abc", "secret": None}}
    storage_options = _resolve_storage_options(["s3://bucket/train.lance/_versions/1.manifest"], download_config)
    assert storage_options == {"key": "abc"}


@pytest.mark.parametrize("streaming", [False, True])
def test_load_lance_streaming_modes(lance_hf_dataset, streaming):
    """Test loading Lance dataset in both streaming and non-streaming modes."""
    from datasets import IterableDataset

    ds = load_dataset(lance_hf_dataset, split="train", streaming=streaming)
    if streaming:
        assert isinstance(ds, IterableDataset)
        items = list(ds)
    else:
        items = list(ds)
    assert len(items) == 4
    assert all("id" in item for item in items)
