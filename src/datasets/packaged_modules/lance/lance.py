import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import pyarrow as pa
from huggingface_hub import HfApi
from huggingface_hub.utils import get_token_to_send

import datasets
from datasets import Audio, Image, Video
from datasets.builder import Key
from datasets.table import table_cast
from datasets.utils.file_utils import is_local_path


if TYPE_CHECKING:
    import lance
    import lance.file

logger = datasets.utils.logging.get_logger(__name__)

MAGIC_BYTES_EXTENSION_AND_FEATURE_TYPES = [
    ("1A 45 DF A3", ".mkv", Video()),
    ("66 74 79 70 69 73 6F 6D", ".mp4", Video()),
    ("66 74 79 70 4D 53 4E 56", ".mp4", Video()),
    ("52 49 46 46", ".avi", Video()),
    ("00 00 01 BA", ".mpeg", Video()),
    ("00 00 01 BA", ".mpeg", Video()),
    ("00 00 01 B3", ".mov", Video()),
    ("89 50 4E 47", ".png", Image()),
    ("FF D8", ".jpg", Image()),
    ("49 49", ".tif", Image()),
    ("47 49 46 38", ".gif", Image()),
    ("52 49 46 46", ".wav", Audio()),
    ("49 44 33", ".mp3", Audio()),
    ("66 4C 61 43", ".flac", Audio()),
]


@dataclass
class LanceConfig(datasets.BuilderConfig):
    """
    BuilderConfig for Lance format.

    Args:
        features: (`Features`, *optional*):
            Cast the data to `features`.
        columns: (`List[str]`, *optional*):
            List of columns to load, the other ones are ignored.
        batch_size: (`int`, *optional*):
            Size of the RecordBatches to iterate on. Default to 256.
        token: (`str`, *optional*):
            Optional HF token to use to download datasets.
    """

    features: Optional[datasets.Features] = None
    columns: Optional[List[str]] = None
    batch_size: Optional[int] = 256
    token: Optional[str] = None


def resolve_dataset_uris(files: List[str]) -> Dict[str, List[str]]:
    dataset_uris = set()
    for file_path in files:
        path = Path(file_path)
        if path.parent.name in {"_transactions", "_indices", "_versions"}:
            dataset_root = path.parent.parent
            dataset_uris.add(str(dataset_root))
    return list(dataset_uris)


def _fix_hf_uri(uri: str) -> str:
    # replace the revision tag from hf uri
    if "@" in uri:
        matched = re.match(r"(hf://.+?)(@[0-9a-f]+)(/.*)", uri)
        if matched:
            uri = matched.group(1) + matched.group(3)
    return uri


def _fix_local_version_file(uri: str) -> str:
    # replace symlinks with real files for _version
    if "/_versions/" in uri and is_local_path(uri):
        path = Path(uri)
        if path.is_symlink():
            data = path.read_bytes()
            path.unlink()
            path.write_bytes(data)
    return uri


def _resolve_hf_token(download_config, token: Optional[Union[bool, str]] = None) -> Optional[str]:
    """Resolve the Hugging Face token to use: the explicit `token` wins, then a token in
    `download_config.storage_options["hf"]` (protocol-specific options override the top-level
    token, like in the fsspec path preparation), then `download_config.token`. The
    `str`/`True`/`False`/`None` semantics are `huggingface_hub`'s (`True` and `None` use the
    ambient token, `False` disables authentication)."""
    hf_storage_options = download_config.storage_options.get("hf") or {}
    token = next((t for t in (token, hf_storage_options.get("token"), download_config.token) if t is not None), None)
    return get_token_to_send(token)


def _resolve_storage_options(
    files: List[str], download_config, token: Optional[Union[bool, str]] = None
) -> Optional[Dict[str, Any]]:
    """Build the storage options to pass to lance for reading `files`.

    Lance's bindings require string values (a `None` value raises a `TypeError`), so `None`
    values are stripped; other values are passed through as-is. Lance's `hf://` object store
    does not fall back to the locally saved Hugging Face token (only the `HF_TOKEN` environment
    variable), so the token must be set explicitly for private repos.
    `download_config.storage_options` can't be passed through as-is: depending on the code path
    it can be missing the `hf` entry entirely (e.g. `get_dataset_split_names()` without a token)
    or contain `{"token": None}` (e.g. `load_dataset(..., streaming=True)`), and the token itself
    can be `True`/`False`, which lance doesn't understand.
    """
    protocol = files[0].split("://", 1)[0] if "://" in files[0] else None
    storage_options = dict(download_config.storage_options.get(protocol) or {}) if protocol else {}
    if protocol == "hf":
        storage_options["token"] = _resolve_hf_token(download_config, token=token)
    return {key: value for key, value in storage_options.items() if value is not None} or None


class Lance(datasets.ArrowBasedBuilder, datasets.builder._CountableBuilderMixin):
    BUILDER_CONFIG_CLASS = LanceConfig
    METADATA_EXTENSIONS = [".idx", ".txn", ".manifest"]
    METADATA_FILE_NAMES = ["latest_version_hint.json"]

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        import lance
        import lance.file

        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        if self.repo_id:
            hf_endpoint = (dl_manager.download_config.storage_options.get("hf") or {}).get("endpoint")
            hf_token = _resolve_hf_token(dl_manager.download_config, token=self.config.token)
            api = HfApi(endpoint=hf_endpoint, token=hf_token if hf_token is not None else False)
            dataset_sha = api.dataset_info(self.repo_id).sha
            if dataset_sha != self.hash:
                raise NotImplementedError(
                    f"lance doesn't support loading other revisions than 'main' yet, but got {self.hash}"
                )
        data_files = dl_manager.download(self.config.data_files)

        # TODO: remove once Lance supports HF links with revisions
        data_files = {split: [_fix_hf_uri(file) for file in files] for split, files in data_files.items()}
        # TODO: remove once Lance supports symlinks for _version files
        data_files = {split: [_fix_local_version_file(file) for file in files] for split, files in data_files.items()}

        splits: list[datasets.SplitGenerator] = []
        for split_name, files in data_files.items():
            storage_options = _resolve_storage_options(files, dl_manager.download_config, token=self.config.token)

            lance_dataset_uris = resolve_dataset_uris(files)
            if lance_dataset_uris:
                lance_datasets = [lance.dataset(uri, storage_options=storage_options) for uri in lance_dataset_uris]
                fragments = [frag for lance_dataset in lance_datasets for frag in lance_dataset.get_fragments()]
                if self.info.features is None:
                    pa_schema = fragments[0]._ds.schema
                    first_row_first_bytes = {}
                    for field in pa_schema:
                        if self.config.columns is not None and field.name not in self.config.columns:
                            continue
                        if pa.types.is_binary(field.type) or pa.types.is_large_binary(field.type):
                            try:
                                first_row_first_bytes[field.name] = (
                                    lance_datasets[0].take_blobs(field.name, [0])[0].read(16)
                                )
                            except ValueError:
                                first_row_first_bytes[field.name] = (
                                    lance_datasets[0].take([0], [field.name]).to_pylist()[0][field.name][:16]
                                )
                splits.append(
                    datasets.SplitGenerator(
                        name=split_name,
                        gen_kwargs={"fragments": fragments, "lance_files_paths": None, "lance_files": None},
                    )
                )
            else:
                lance_files = [
                    lance.file.LanceFileReader(file, storage_options=storage_options, columns=self.config.columns)
                    for file in files
                ]
                if self.info.features is None:
                    pa_schema = lance_files[0].metadata().schema
                    first_row_first_bytes = {
                        field_name: value[:16]
                        for field_name, value in lance_files[0].take_rows([0]).to_table().to_pylist()[0].items()
                        if isinstance(value, bytes)
                    }
                splits.append(
                    datasets.SplitGenerator(
                        name=split_name,
                        gen_kwargs={"fragments": None, "lance_files_paths": files, "lance_files": lance_files},
                    )
                )
            if self.info.features is None:
                if self.config.columns:
                    fields = [
                        pa_schema.field(name) for name in self.config.columns if pa_schema.get_field_index(name) != -1
                    ]
                    pa_schema = pa.schema(fields)
                features = datasets.Features.from_arrow_schema(pa_schema)
                for field_name, first_bytes in first_row_first_bytes.items():
                    for magic_bytes_hex, _, feature_type in MAGIC_BYTES_EXTENSION_AND_FEATURE_TYPES:
                        magic_bytes = bytes.fromhex(magic_bytes_hex)
                        if magic_bytes in first_bytes[: len(magic_bytes) * 2]:  # allow some padding
                            features[field_name] = feature_type
                            break
                self.info.features = features

        return splits

    def _cast_table(self, pa_table: pa.Table) -> pa.Table:
        if self.info.features is not None:
            # more expensive cast to support nested features with keys in a different order
            # allows str <-> int/float or str to Audio for example
            pa_table = table_cast(pa_table, self.info.features.arrow_schema)
        return pa_table

    def _generate_shards(
        self,
        fragments: Optional[List["lance.LanceFragment"]],
        lance_files_paths: Optional[list[str]],
        lance_files: Optional[List["lance.file.LanceFileReader"]],
    ):
        if fragments:
            for fragment in fragments:
                paths = [data_file.path for data_file in fragment.metadata.data_files()]
                yield paths[0] if len(paths) == 1 else {"fragment_data_files": paths}
        else:
            yield from lance_files_paths

    def _generate_num_examples(
        self,
        fragments: Optional[List["lance.LanceFragment"]],
        lance_files_paths: Optional[list[str]],
        lance_files: Optional[List["lance.file.LanceFileReader"]],
    ):
        if fragments:
            for fragment in fragments:
                yield fragment.count_rows()
        else:
            for lance_file in lance_files:
                yield lance_file.num_rows()

    def _generate_tables(
        self,
        fragments: Optional[List["lance.LanceFragment"]],
        lance_files_paths: Optional[list[str]],
        lance_files: Optional[List["lance.file.LanceFileReader"]],
    ):
        if fragments:
            for frag_idx, fragment in enumerate(fragments):
                for batch_idx, batch in enumerate(
                    fragment.to_batches(
                        columns=self.config.columns, batch_size=self.config.batch_size, blob_handling="all_binary"
                    )
                ):
                    table = pa.Table.from_batches([batch])
                    yield Key(frag_idx, batch_idx), self._cast_table(table)
        else:
            for file_idx, lance_file in enumerate(lance_files):
                for batch_idx, batch in enumerate(lance_file.read_all(batch_size=self.config.batch_size).to_batches()):
                    table = pa.Table.from_batches([batch])
                    yield Key(file_idx, batch_idx), self._cast_table(table)
