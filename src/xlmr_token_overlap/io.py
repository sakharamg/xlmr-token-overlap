"""Input preparation and corpus loading."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from .constants import (
    FLORES_SHA256,
    FLORES_SPLIT_SIZES,
    FLORES_URL,
    LANGUAGE_CODES,
    XLMR_REPOSITORY,
    XLMR_REVISION,
    XLMR_TOKENIZER_SHA256,
    XLMR_TOKENIZER_URL,
)


@dataclass(frozen=True, slots=True)
class Record:
    condition: str
    language_code: str
    text: str
    example_id: str
    split: str


def portable_path(path: Path) -> str:
    """Prefer a reproducible path relative to the current project."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256(destination) == expected_sha256:
        print(f"[reuse] {destination}")
        return

    temporary = destination.with_name(destination.name + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "xlmr-token-overlap/0.1 (+reproducible-research)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        actual = sha256(temporary)
        if actual != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {actual}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[download] {destination}")


def _extract_selected_flores(archive: Path, target: Path) -> None:
    wanted = {PurePosixPath("flores200_dataset/README")}
    for split in FLORES_SPLIT_SIZES:
        wanted.add(PurePosixPath(f"flores200_dataset/metadata_{split}.tsv"))
        for code in LANGUAGE_CODES:
            wanted.add(PurePosixPath(f"flores200_dataset/{split}/{code}.{split}"))

    with tarfile.open(archive, "r:gz") as bundle:
        members: dict[PurePosixPath, tarfile.TarInfo] = {}
        for member in bundle.getmembers():
            relative = PurePosixPath(member.name.lstrip("./"))
            if relative in wanted:
                members[relative] = member
        missing = sorted(wanted - members.keys(), key=str)
        if missing:
            raise RuntimeError(
                "FLORES archive is missing required members: "
                + ", ".join(str(item) for item in missing[:5])
            )

        for relative in sorted(wanted, key=str):
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe archive member: {relative}")
            member = members[relative]
            if not member.isfile():
                raise RuntimeError(f"Expected regular file: {relative}")
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read archive member: {relative}")
            destination = target.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            # The selected FLORES payload is only ~9 MB. Reading each member
            # fully lets us verify tar metadata before an atomic replacement,
            # so an interrupted or short write can never look like valid data.
            with source:
                payload = source.read()
            if len(payload) != member.size:
                raise RuntimeError(
                    f"Short archive read for {relative}: expected {member.size}, got {len(payload)}"
                )
            temporary = destination.with_name(destination.name + ".part")
            try:
                with temporary.open("wb") as output:
                    written = output.write(payload)
                    output.flush()
                    os.fsync(output.fileno())
                if written != member.size or temporary.stat().st_size != member.size:
                    raise RuntimeError(f"Short archive write for {relative}")
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
    print(f"[extract] {target / 'flores200_dataset'}")


def prepare_flores(source_dir: Path) -> dict[str, Path]:
    """Download pinned sources and safely extract the selected FLORES files."""

    downloads = source_dir / "downloads"
    archive = downloads / "flores200_dataset.tar.gz"
    tokenizer_json = source_dir / "tokenizers" / "xlm-roberta-base" / "tokenizer.json"
    _download(FLORES_URL, archive, FLORES_SHA256)
    _download(XLMR_TOKENIZER_URL, tokenizer_json, XLMR_TOKENIZER_SHA256)
    _extract_selected_flores(archive, source_dir)

    source_manifest = {
        "flores": {
            "url": FLORES_URL,
            "sha256": FLORES_SHA256,
            "splits": FLORES_SPLIT_SIZES,
        },
        "tokenizer": {
            "repository": XLMR_REPOSITORY,
            "revision": XLMR_REVISION,
            "url": XLMR_TOKENIZER_URL,
            "sha256": XLMR_TOKENIZER_SHA256,
        },
        "languages": list(LANGUAGE_CODES),
    }
    (source_dir / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "flores_root": source_dir / "flores200_dataset",
        "tokenizer_json": tokenizer_json,
    }


def load_flores(flores_root: Path, splits: Iterable[str] = ("dev", "devtest")) -> tuple[list[Record], dict]:
    """Load all selected languages and strictly verify parallel alignment."""

    selected_splits = tuple(splits)
    unknown_splits = set(selected_splits) - FLORES_SPLIT_SIZES.keys()
    if unknown_splits:
        raise ValueError(f"Unknown FLORES splits: {sorted(unknown_splits)}")

    records: list[Record] = []
    files: list[dict] = []
    alignment: dict[str, list[str]] = {}
    for split in selected_splits:
        expected = FLORES_SPLIT_SIZES[split]
        reference_ids = [f"{split}:{index:04d}" for index in range(expected)]
        alignment[split] = reference_ids
        for code in LANGUAGE_CODES:
            path = flores_root / split / f"{code}.{split}"
            if not path.is_file():
                raise FileNotFoundError(f"Missing FLORES input: {path}")
            texts = path.read_text(encoding="utf-8").splitlines()
            if len(texts) != expected:
                raise ValueError(
                    f"Alignment failure for {code}/{split}: expected {expected} lines, got {len(texts)}"
                )
            if any(not text for text in texts):
                first = next(index for index, text in enumerate(texts) if not text)
                raise ValueError(f"Empty FLORES line at {code}/{split}/{first}")
            files.append(
                {
                    "language_code": code,
                    "split": split,
                    "path": portable_path(path),
                    "sha256": sha256(path),
                    "examples": len(texts),
                }
            )
            records.extend(
                Record("flores", code, text, example_id, split)
                for text, example_id in zip(texts, reference_ids, strict=True)
            )

    provenance = {
        "dataset": "FLORES-200",
        "condition": "flores",
        "root": portable_path(flores_root),
        "splits": {split: FLORES_SPLIT_SIZES[split] for split in selected_splits},
        "examples_per_language": sum(FLORES_SPLIT_SIZES[split] for split in selected_splits),
        "parallel_alignment": "same split and line position across all languages",
        "files": files,
    }
    return records, provenance


def load_jsonl(path: Path) -> tuple[list[Record], dict]:
    """Load condition-aware rows for Pass 2/3 without pooling conditions.

    Required fields are `language_code` and `text`. Optional fields are
    `condition`, `example_id`, and `split`. Each condition is analyzed into a
    separate output directory by the caller.
    """

    records: list[Record] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            code = str(row["language_code"])
            if code not in LANGUAGE_CODES:
                raise ValueError(f"Unsupported language_code at line {line_number}: {code}")
            text = str(row["text"])
            if not text:
                raise ValueError(f"Empty text at line {line_number}")
            records.append(
                Record(
                    condition=str(row.get("condition", "default")),
                    language_code=code,
                    text=text,
                    example_id=str(row.get("example_id", f"row:{line_number:08d}")),
                    split=str(row.get("split", "unspecified")),
                )
            )
    if not records:
        raise ValueError(f"No records found in {path}")
    return records, {
        "dataset": "JSONL interchange",
        "path": portable_path(path),
        "sha256": sha256(path),
        "row_count": len(records),
        "schema": list(asdict(records[0]).keys()),
    }
