"""Extract pinned text columns into local ``<source>/<revision>/text.parquet``.

Use HTTP ranges to omit parquet audio and stop JSONL at declared budgets.
Write unchanged rows plus ``extraction.json``; normalization stays in the loader."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit

from akanlm.audit.sources import CorpusSource
from akanlm.utils.logging import get_logger
from akanlm.utils.paths import file_sha256

logger = get_logger(__name__)

# Keep the existing extraction contract readable. The optional URL-domain
# summary is supplemental metadata, not a change to the text-table format.
PROTOCOL = "hf_column_extraction_v1"
TABLE = "text.parquet"
RECORD = "extraction.json"
BATCH_ROWS = 10_000


def extract_root() -> Path:
    return Path(os.environ.get("AKANLM_EXTRACT_ROOT", "data/raw/corpora"))


def extraction_dir(source: CorpusSource, root: Path | None = None) -> Path:
    return (root or extract_root()) / source.name / str(source.hf_revision)


def declaration(source: CorpusSource) -> dict:
    """The declaration fields an extraction depends on."""
    return {"hf_id": source.hf_id, "hf_revision": source.hf_revision, "hf_files": source.hf_files,
            "hf_columns": source.hf_columns, "text_column": source.text_column}


def implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def url_domain_summary(rows: Iterator[dict], column: str = "url", *, top: int = 100) -> dict:
    """Report hostname counts, without inferring registrable domains.

    A fixed mC4 prefix is not a random sample."""
    domains: Counter[str] = Counter()
    urls_seen = invalid_or_missing = 0
    for row in rows:
        value = row.get(column)
        if not isinstance(value, str) or not value.strip():
            invalid_or_missing += 1
            continue
        urls_seen += 1
        hostname = urlsplit(value).hostname
        if not hostname:
            invalid_or_missing += 1
            continue
        domains[hostname.lower().rstrip(".")] += 1
    return {
        "url_column": column,
        "urls_seen": urls_seen,
        "invalid_or_missing_urls": invalid_or_missing,
        "unique_hostnames": len(domains),
        "top_hostnames": [
            {"hostname": hostname, "documents": count}
            for hostname, count in domains.most_common(top)
        ],
    }


def resolve_files(source: CorpusSource, fs) -> list[dict]:
    """Repository files matching ``hf_files`` at the pinned revision, in sorted order."""
    prefix = f"datasets/{source.hf_id}@{source.hf_revision}/"
    files: dict[str, int] = {}
    for pattern in source.hf_files or []:
        matches = sorted(fs.glob(prefix + pattern))
        if not matches:
            raise FileNotFoundError(f"{source.name}: no files match {pattern!r} at {source.hf_revision}")
        for path in matches:
            files.setdefault(path[len(prefix):], fs.info(path)["size"])
    return [{"path": path, "size": size} for path, size in files.items()]


def _rows(fs, path: str, columns: list[str]) -> Iterator[dict]:
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        with fs.open(path, "rb", block_size=4 * 2**20) as handle:
            for batch in pq.ParquetFile(handle).iter_batches(columns=columns, batch_size=BATCH_ROWS):
                yield from batch.to_pylist()
    elif path.endswith((".json", ".jsonl", ".json.gz", ".jsonl.gz")):
        with fs.open(path, "rb", block_size=8 * 2**20) as raw:
            binary = gzip.GzipFile(fileobj=raw) if path.endswith(".gz") else raw
            for line in io.TextIOWrapper(binary, encoding="utf-8"):
                if line.strip():
                    row = json.loads(line)
                    yield {column: row.get(column) for column in columns}
    else:
        raise ValueError(f"unsupported extraction format: {path}")


def _satisfies(record: dict, max_rows: int | None, max_chars: int | None) -> bool:
    if record["exhausted"]:
        return True
    rows_ok = max_rows is not None and record["budget"]["rows"] is not None and record["budget"]["rows"] >= max_rows
    chars_ok = max_chars is not None and record["budget"]["chars"] is not None and record["budget"]["chars"] >= max_chars
    return rows_ok or chars_ok


def extract(source: CorpusSource, root: Path | None = None, *, max_rows: int | None = None,
            max_chars: int | None = None, fs=None) -> dict:
    """Extract declared columns up to normalized row/character budgets.

    Keep the last document whole. Reuse only matching extractions covering the budget;
    refuse to overwrite incompatible output."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from akanlm.corpora.normalize import normalization_spec, normalize

    if not source.hf_files or not source.hf_revision or len(source.hf_revision) != 40:
        raise ValueError(f"{source.name}: extraction needs hf_files and a 40-character hf_revision")
    if source.group_by and (max_rows is not None or max_chars is not None):
        raise ValueError(f"{source.name}: a row or character budget cannot apply to group_by documents")
    destination = extraction_dir(source, root)
    if destination.exists():
        record = read_record(source, root)
        if not _satisfies(record, max_rows, max_chars):
            raise FileExistsError(f"{destination} holds a smaller budget; extract into a fresh root")
        logger.info("%s: reusing extraction at %s", source.name, destination)
        return record
    if fs is None:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()

    files = resolve_files(source, fs)
    rules = source.effective_rules()
    prefix = f"datasets/{source.hf_id}@{source.hf_revision}/"
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    rows = emitted = chars = 0
    writer = None
    batch: list[dict] = []
    domain_rows: list[dict] | None = [] if "url" in source.hf_columns else None
    reached = False

    def flush():
        nonlocal writer
        if not batch:
            return
        table = pa.Table.from_pylist(batch)
        if writer is None:
            # An all-null column in the first batch must not fix the file's type.
            schema = pa.schema([field.with_type(pa.string()) if pa.types.is_null(field.type) else field
                                for field in table.schema])
            writer = pq.ParquetWriter(temporary / TABLE, schema)
        writer.write_table(table.cast(writer.schema))
        batch.clear()

    for entry in files:
        logger.info("%s: reading %s (%d bytes on the Hub)", source.name, entry["path"], entry["size"])
        for row in _rows(fs, prefix + entry["path"], source.hf_columns):
            batch.append(row)
            if domain_rows is not None:
                domain_rows.append(row)
            rows += 1
            text = row.get(source.text_column)
            if isinstance(text, str) and text and (max_rows is not None or max_chars is not None):
                normalized = normalize(text, rules)[0]
                if normalized:
                    emitted += 1
                    chars += len(normalized)
            if len(batch) >= BATCH_ROWS:
                flush()
            if (max_rows is not None and emitted >= max_rows) or (max_chars is not None and chars >= max_chars):
                reached = True
                break
        if reached:
            break
    flush()
    if writer is None:
        raise ValueError(f"{source.name}: extraction found no rows")
    writer.close()
    record = {
        "protocol": PROTOCOL, "declaration": declaration(source), "files": files,
        "budget": {"rows": max_rows, "chars": max_chars}, "rows": rows,
        "budget_normalization": normalization_spec(rules) if reached else None,
        "normalized_chars_counted": chars if reached else None,
        "exhausted": not reached, "table_sha256": file_sha256(temporary / TABLE),
        "implementation_sha256": implementation_sha256(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if domain_rows is not None:
        record["url_domain_summary"] = url_domain_summary(iter(domain_rows))
    if max_chars is not None and not reached:
        raise ValueError(f"{source.name}: files hold {chars:,} normalized characters, "
                         f"fewer than the {max_chars:,} the mixtures budget")
    (temporary / RECORD).write_text(json.dumps(record, indent=2))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(destination)
    logger.info("%s: wrote %d rows to %s", source.name, rows, destination)
    return record


def read_record(source: CorpusSource, root: Path | None = None) -> dict:
    directory = extraction_dir(source, root)
    path = directory / RECORD
    if not path.exists():
        raise FileNotFoundError(
            f"{source.name}: no extraction at {directory}. Extract it where the Hub is "
            "reachable with akanlm.corpora.extract.extract (see README.md).")
    record = json.loads(path.read_text())
    if record.get("protocol") != PROTOCOL or record.get("declaration") != declaration(source):
        raise ValueError(f"{source.name}: {directory} was extracted under a different declaration")
    return record


def read_rows(source: CorpusSource, root: Path | None = None, *, verify: bool = True) -> Iterator[dict]:
    """Yield local rows, or newline-joined documents for ``group_by`` sources.

    Groups sort ascending; text follows ``order_by`` with group/row-count metadata."""
    import pyarrow.parquet as pq

    record = read_record(source, root)
    table_path = extraction_dir(source, root) / TABLE
    if verify and file_sha256(table_path) != record["table_sha256"]:
        raise ValueError(f"{source.name}: {table_path} does not match its extraction record")
    parquet = pq.ParquetFile(table_path)
    if not source.group_by:
        for batch in parquet.iter_batches(batch_size=BATCH_ROWS):
            yield from batch.to_pylist()
        return
    table = parquet.read(columns=[source.group_by, source.order_by, source.text_column])
    table = table.sort_by([(source.group_by, "ascending"), (source.order_by, "ascending")])
    groups = table.column(source.group_by).to_pylist()
    texts = table.column(source.text_column).to_pylist()
    start = 0
    for end in range(1, len(groups) + 1):
        if end < len(groups) and groups[end] == groups[start]:
            continue
        lines = [text for text in texts[start:end] if isinstance(text, str) and text.strip()]
        if lines:
            yield {source.group_by: groups[start], source.text_column: "\n".join(lines),
                   "rows_joined": len(lines)}
        start = end
