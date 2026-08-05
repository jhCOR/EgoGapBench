#!/usr/bin/env python3
"""Download every image EgoGapBench needs and normalise the paths inside the JSONL files.

Two image sources are handled:

  coco       MS-COCO images referenced by ``Data/benchmark/egogapbench/*.bundles.jsonl``
             and ``Data/traindata/egogapbench_split.jsonl``   ->  ``Data/images/coco/``
  wikimedia  Wikimedia Commons images listed in ``Data/data_prepare/wikimedia_manifest.jsonl``
                                                              ->  ``Data/images/wikimedia/``

Nothing is hard-coded per image: the work list is derived from the JSONL files themselves,
so adding a bundle/manifest row is enough to make its image part of the download.

Usage
-----
    python Data/data_prepare/download_image.py                 # coco + wikimedia + fix-paths
    python Data/data_prepare/download_image.py coco
    python Data/data_prepare/download_image.py wikimedia
    python Data/data_prepare/download_image.py fix-paths       # rewrite JSONL paths only

Every path can be overridden from the command line; the defaults are resolved relative to
this file's repository so the script works from any working directory and on any machine.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence
from urllib.parse import unquote, urlparse

import requests

LOG = logging.getLogger("egogapbench.download")

# --- repository layout -------------------------------------------------------------------
# .../EgoGapBench/Data/data_prepare/download_image.py -> parents[2] == repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "Data"

IMAGES_REL = Path("Data/images")  # written into the JSONL files (repo-root relative)
COCO_SUBDIR = "coco"
WIKI_SUBDIR = "wikimedia"

# JSONL sources of the work list, relative to the data directory.
COCO_BUNDLE_GLOB = "benchmark/egogapbench/*.bundles.jsonl"
COCO_TRAIN_SPLIT = "traindata/egogapbench_split.jsonl"
# One row per Commons image: {"image_id", "file_name", "source", "license"}.
WIKI_MANIFEST_GLOB = "data_prepare/wikimedia_manifest.jsonl"

# --- remote endpoints --------------------------------------------------------------------
COCO_IMAGE_URL = "http://images.cocodataset.org/{split}/{file_name}"
COCO_SPLITS: tuple[str, ...] = ("train2017", "val2017")
COCO_ANN_FILES = ("instances_train2017.json", "instances_val2017.json")

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
COMMONS_API_BATCH = 50  # titles per query, the anonymous API limit

# Wikimedia requires a descriptive User-Agent. Project identity only, no personal data.
DEFAULT_USER_AGENT = (
    "EgoGapBench-DataPrep/1.0 (https://github.com/jhCOR/EgoGapBench) python-requests"
)

JPEG_QUALITY = 95
DEFAULT_WORKERS = 8
DEFAULT_RETRIES = 5
DEFAULT_TIMEOUT = 60

# upload.wikimedia.org throttles bulk clients hard (HTTP 429). Stay deliberately polite:
# few workers, a pause between requests, and long waits when we are asked to slow down.
DEFAULT_WIKI_WORKERS = 2
DEFAULT_WIKI_DELAY = 0.5
THROTTLE_BACKOFF = (5, 15, 30, 60, 120)


# ==========================================================================================
# small helpers
# ==========================================================================================
def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc


def rewrite_jsonl(path: Path, transform: Callable[[dict], bool], *, dry_run: bool = False) -> int:
    """Apply ``transform`` to every row in place. Returns the number of changed rows."""
    rows = list(iter_jsonl(path))
    changed = sum(1 for row in rows if transform(row))
    if not changed or dry_run:
        return changed
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return changed


def make_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    return session


def run_parallel(jobs: Sequence[Callable[[], object]], workers: int) -> list:
    """Run ``jobs`` with a thread pool, preserving order. Exceptions become ``None``."""
    if not jobs:
        return []
    workers = max(1, min(workers, len(jobs)))
    results: list = [None] * len(jobs)
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(job): i for i, job in enumerate(jobs)}
        for future in cf.as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad image must not kill the run
                LOG.error("job %d failed: %s", index, exc)
    return results


def retry_after_seconds(response: requests.Response | None, attempt: int) -> float:
    """How long to wait before retrying a throttled request."""
    if response is not None:
        header = response.headers.get("Retry-After")
        if header:
            try:
                return max(1.0, float(header))
            except ValueError:
                pass
    return float(THROTTLE_BACKOFF[min(attempt, len(THROTTLE_BACKOFF) - 1)])


def fetch_bytes(
    session: requests.Session,
    url: str,
    *,
    retries: int = DEFAULT_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    allow_404: bool = False,
) -> bytes | None:
    """GET ``url`` with backoff. ``None`` when the server answers 404 and ``allow_404``."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        wait = float(2**attempt)
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 404 and allow_404:
                return None
            if response.status_code in (429, 503):
                wait = retry_after_seconds(response, attempt)
                last_exc = requests.HTTPError(f"HTTP {response.status_code}", response=response)
                LOG.debug("throttled (%s), waiting %.0fs: %s", response.status_code, wait, url)
            else:
                response.raise_for_status()
                return response.content
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and 400 <= status < 500:
                raise
            last_exc = exc
        except requests.RequestException as exc:
            last_exc = exc
        if attempt < retries - 1:
            time.sleep(wait)
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_exc}")


def write_image(payload: bytes, dest: Path) -> None:
    """Persist ``payload`` at ``dest``, transcoding to JPEG when the suffix says so."""
    from PIL import Image  # imported lazily so `fix-paths` needs no Pillow

    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(payload)) as img:
        img.load()
        needs_jpeg = dest.suffix.lower() in {".jpg", ".jpeg"}
        if needs_jpeg and img.format != "JPEG":
            rgb = img.convert("RGB")
            fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), suffix=dest.suffix)
            os.close(fd)
            tmp = Path(tmp_name)
            try:
                rgb.save(tmp, format="JPEG", quality=JPEG_QUALITY)
                tmp.replace(dest)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
            return
    fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), suffix=dest.suffix)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


@dataclass
class Stats:
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0

    def add(self, other: "Stats") -> None:
        self.downloaded += other.downloaded
        self.skipped += other.skipped
        self.failed += other.failed

    def __str__(self) -> str:
        return f"downloaded={self.downloaded} skipped={self.skipped} failed={self.failed}"


# ==========================================================================================
# COCO
# ==========================================================================================
def collect_coco_file_names(data_dir: Path) -> list[str]:
    """Every COCO file name referenced by the benchmark bundles and the training split."""
    names: set[str] = set()

    for bundle in sorted(data_dir.glob(COCO_BUNDLE_GLOB)):
        for row in iter_jsonl(bundle):
            file_name = row.get("file_name")
            if file_name:
                names.add(Path(file_name).name)
        LOG.debug("scanned %s", bundle.relative_to(data_dir))

    split = data_dir / COCO_TRAIN_SPLIT
    if split.is_file():
        for row in iter_jsonl(split):
            for image in row.get("images") or []:
                names.add(Path(image).name)
        LOG.debug("scanned %s", split.relative_to(data_dir))
    else:
        LOG.warning("training split not found: %s", split)

    return sorted(names)


def coco_urls_from_api(ann_dir: Path) -> dict[str, str]:
    """Map ``file_name -> coco_url`` through the COCO API (needs pycocotools + annotations)."""
    from pycocotools.coco import COCO  # noqa: PLC0415 - optional dependency

    mapping: dict[str, str] = {}
    for ann_name in COCO_ANN_FILES:
        ann_path = ann_dir / ann_name
        if not ann_path.is_file():
            LOG.debug("annotation file missing, skipped: %s", ann_path)
            continue
        LOG.info("loading COCO annotations: %s", ann_path)
        coco = COCO(str(ann_path))
        for img in coco.loadImgs(coco.getImgIds()):
            url = img.get("coco_url")
            if url:
                mapping[img["file_name"]] = url
    return mapping


def resolve_coco_urls(file_names: Sequence[str], ann_dir: Path | None) -> dict[str, str]:
    """Resolve download URLs, preferring the COCO API when annotations are available.

    Falls back to the public bucket layout (which is exactly what ``coco_url`` points at),
    so a fresh clone does not have to fetch the multi-hundred-MB annotation archive.
    """
    if ann_dir is not None and ann_dir.is_dir():
        try:
            mapping = coco_urls_from_api(ann_dir)
        except ImportError:
            LOG.warning("pycocotools is not installed - falling back to direct image URLs")
        else:
            missing = [name for name in file_names if name not in mapping]
            if not missing:
                LOG.info("resolved %d image URLs through the COCO API", len(file_names))
                return {name: mapping[name] for name in file_names}
            LOG.warning(
                "COCO API resolved %d/%d names (%d missing) - using direct URLs for the rest",
                len(file_names) - len(missing), len(file_names), len(missing),
            )
            return {name: mapping[name] for name in file_names if name in mapping}
    else:
        LOG.info("no COCO annotation directory - using the public image URL layout")
    return {}


def download_coco_image(
    session: requests.Session,
    file_name: str,
    dest_dir: Path,
    *,
    url: str | None,
    splits: Sequence[str],
    overwrite: bool,
    retries: int,
    timeout: int,
) -> Stats:
    dest = dest_dir / file_name
    if dest.is_file() and dest.stat().st_size > 0 and not overwrite:
        return Stats(skipped=1)

    candidates = [url] if url else [
        COCO_IMAGE_URL.format(split=split, file_name=file_name) for split in splits
    ]
    for candidate in candidates:
        payload = fetch_bytes(
            session, candidate, retries=retries, timeout=timeout, allow_404=True
        )
        if payload is None:
            continue
        write_image(payload, dest)
        return Stats(downloaded=1)

    LOG.error("COCO image not found in %s: %s", "/".join(splits), file_name)
    return Stats(failed=1)


def cmd_coco(args: argparse.Namespace) -> Stats:
    data_dir: Path = args.data_dir
    dest_dir: Path = args.images_dir / COCO_SUBDIR
    file_names = collect_coco_file_names(data_dir)
    if not file_names:
        LOG.warning("no COCO images referenced by the JSONL files")
        return Stats()

    LOG.info("COCO: %d unique images -> %s", len(file_names), dest_dir)
    if args.dry_run:
        return Stats(skipped=len(file_names))

    urls = resolve_coco_urls(file_names, args.coco_ann_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    session = make_session(args.user_agent)

    jobs = [
        (lambda name=name: download_coco_image(
            session, name, dest_dir,
            url=urls.get(name), splits=args.coco_splits,
            overwrite=args.overwrite, retries=args.retries, timeout=args.timeout,
        ))
        for name in file_names
    ]
    total = Stats()
    for index, result in enumerate(run_parallel(jobs, args.workers)):
        total.add(result if isinstance(result, Stats) else Stats(failed=1))
        if (index + 1) % 100 == 0:
            LOG.info("COCO progress: %d/%d (%s)", index + 1, len(jobs), total)
    LOG.info("COCO done: %s", total)
    return total


# ==========================================================================================
# Wikimedia Commons
# ==========================================================================================
@dataclass
class WikiTarget:
    image_id: int | str
    title: str          # "File:Example.jpg"
    dest_name: str      # "manual_910013.jpg"
    source: str
    license: str


def commons_title_from_url(source: str) -> str | None:
    """Extract the ``File:...`` title from a Commons file-page URL."""
    parsed = urlparse(source)
    if "wikimedia.org" not in parsed.netloc and "wikipedia.org" not in parsed.netloc:
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2 and parts[0] == "wiki":
        title = unquote("/".join(parts[1:])).replace("_", " ")
        return title if title.lower().startswith("file:") else None
    return None


def collect_wiki_targets(data_dir: Path) -> list[WikiTarget]:
    """Every Commons image listed in the Wikimedia manifest."""
    targets: dict[str, WikiTarget] = {}
    manifests = sorted(data_dir.glob(WIKI_MANIFEST_GLOB))
    if not manifests:
        LOG.warning("no Wikimedia manifest matched %s", WIKI_MANIFEST_GLOB)

    for manifest in manifests:
        for row in iter_jsonl(manifest):
            source = row.get("source")
            image_id = row.get("image_id")
            if not source or image_id is None:
                continue
            title = commons_title_from_url(source)
            if not title:
                LOG.warning("unsupported source URL, skipped: %s", source)
                continue
            dest_name = Path(row.get("file_name") or f"manual_{image_id}.jpg").name
            targets[dest_name] = WikiTarget(
                image_id=image_id,
                title=title,
                dest_name=dest_name,
                source=source,
                license=row.get("license") or "",
            )
        LOG.debug("scanned %s", manifest.relative_to(data_dir))
    return sorted(targets.values(), key=lambda t: str(t.dest_name))


def query_commons_imageinfo(
    session: requests.Session,
    titles: Sequence[str],
    *,
    retries: int,
    timeout: int,
) -> dict[str, dict]:
    """Batch ``action=query&prop=imageinfo`` and return ``normalised title -> imageinfo``."""
    info: dict[str, dict] = {}
    for start in range(0, len(titles), COMMONS_API_BATCH):
        batch = titles[start:start + COMMONS_API_BATCH]
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "imageinfo",
            "iiprop": "url|mime|extmetadata",
            "titles": "|".join(batch),
        }
        last_exc: Exception | None = None
        payload = None
        for attempt in range(retries):
            try:
                response = session.get(COMMONS_API, params=params, timeout=timeout)
                response.raise_for_status()
                payload = response.json()
                break
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                if attempt < retries - 1:
                    time.sleep(2**attempt)
        if payload is None:
            raise RuntimeError(f"Commons API query failed: {last_exc}")

        query = payload.get("query", {})
        # The API normalises titles (underscores -> spaces); map them back to what we asked.
        alias = {n["to"]: n["from"] for n in query.get("normalized", [])}
        for page in query.get("pages", []):
            title = page.get("title", "")
            requested = alias.get(title, title)
            images = page.get("imageinfo") or []
            if page.get("missing") or not images:
                LOG.error("Commons page has no image info: %s", requested)
                continue
            info[requested] = images[0]
        LOG.info("Commons API: %d/%d titles resolved", len(info), len(titles))
    return info


def commons_credit(target: WikiTarget, imageinfo: dict) -> dict:
    """Attribution record for the CC-BY style licences used by the manifest."""
    meta = imageinfo.get("extmetadata") or {}

    def field(key: str) -> str:
        value = (meta.get(key) or {}).get("value")
        return value if isinstance(value, str) else ""

    return {
        "file": target.dest_name,
        "image_id": target.image_id,
        "source_page": target.source,
        "file_url": imageinfo.get("url", ""),
        "license": target.license or field("LicenseShortName"),
        "license_url": field("LicenseUrl"),
        "artist_html": field("Artist"),
    }


def download_wiki_image(
    session: requests.Session,
    target: WikiTarget,
    imageinfo: dict,
    dest_dir: Path,
    *,
    overwrite: bool,
    retries: int,
    timeout: int,
    delay: float,
) -> Stats:
    dest = dest_dir / target.dest_name
    if dest.is_file() and dest.stat().st_size > 0 and not overwrite:
        return Stats(skipped=1)
    url = imageinfo.get("url")
    if not url:
        LOG.error("no file URL for %s", target.title)
        return Stats(failed=1)
    if delay > 0:
        time.sleep(delay)
    payload = fetch_bytes(session, url, retries=retries, timeout=timeout)
    write_image(payload, dest)  # transcodes png/jpeg originals into the expected .jpg
    return Stats(downloaded=1)


def cmd_wikimedia(args: argparse.Namespace) -> Stats:
    dest_dir: Path = args.images_dir / WIKI_SUBDIR
    targets = collect_wiki_targets(args.data_dir)
    if not targets:
        return Stats()

    LOG.info("Wikimedia: %d unique images -> %s", len(targets), dest_dir)
    if args.dry_run:
        return Stats(skipped=len(targets))

    dest_dir.mkdir(parents=True, exist_ok=True)
    session = make_session(args.user_agent)
    info = query_commons_imageinfo(
        session, [t.title for t in targets], retries=args.retries, timeout=args.timeout
    )

    resolved = [(t, info[t.title]) for t in targets if t.title in info]
    total = Stats(failed=len(targets) - len(resolved))
    jobs = [
        (lambda t=t, i=i: download_wiki_image(
            session, t, i, dest_dir,
            overwrite=args.overwrite, retries=args.retries, timeout=args.timeout,
            delay=args.wiki_delay,
        ))
        for t, i in resolved
    ]
    for result in run_parallel(jobs, args.wiki_workers):
        total.add(result if isinstance(result, Stats) else Stats(failed=1))

    credits_path = dest_dir / "CREDITS.json"
    credits_path.write_text(
        json.dumps([commons_credit(t, i) for t, i in resolved], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    LOG.info("Wikimedia done: %s (attribution written to %s)", total, credits_path)
    return total


# ==========================================================================================
# JSONL path normalisation
# ==========================================================================================
def cmd_fix_paths(args: argparse.Namespace) -> Stats:
    """Rewrite absolute media paths in the JSONL files to repo-root relative ones."""
    data_dir: Path = args.data_dir
    coco_rel = args.images_rel / COCO_SUBDIR
    wiki_rel = args.images_rel / WIKI_SUBDIR
    total = Stats()

    split = data_dir / COCO_TRAIN_SPLIT
    if split.is_file():
        def fix_split(row: dict) -> bool:
            images = row.get("images")
            if not images:
                return False
            fixed = [str(coco_rel / Path(image).name) for image in images]
            if fixed == images:
                return False
            row["images"] = fixed
            return True

        changed = rewrite_jsonl(split, fix_split, dry_run=args.dry_run)
        LOG.info("%s: %d rows rewritten -> %s/", split.relative_to(data_dir), changed, coco_rel)
        total.downloaded += changed
    else:
        LOG.warning("training split not found: %s", split)

    for manifest in sorted(data_dir.glob(WIKI_MANIFEST_GLOB)):
        def fix_manifest(row: dict) -> bool:
            current = row.get("file_name")
            if not current:
                return False
            fixed = str(wiki_rel / Path(current).name)
            if fixed == current:
                return False
            row["file_name"] = fixed
            return True

        changed = rewrite_jsonl(manifest, fix_manifest, dry_run=args.dry_run)
        LOG.info("%s: %d rows rewritten -> %s/", manifest.relative_to(data_dir), changed, wiki_rel)
        total.downloaded += changed

    return total


# ==========================================================================================
# CLI
# ==========================================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "target", nargs="?", default="all", choices=("all", "coco", "wikimedia", "fix-paths"),
        help="what to run (default: all)",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="EgoGapBench Data directory (default: %(default)s)")
    parser.add_argument("--images-dir", type=Path, default=None,
                        help="image output directory (default: <data-dir>/images)")
    parser.add_argument("--images-rel", type=Path, default=IMAGES_REL,
                        help="path prefix written into the JSONL files (default: %(default)s)")
    parser.add_argument("--coco-ann-dir", type=Path,
                        default=os.environ.get("COCO_ANN_DIR") or None,
                        help="directory holding instances_{train,val}2017.json; when present "
                             "URLs are resolved with the COCO API instead of the URL layout")
    parser.add_argument("--coco-splits", nargs="+", default=list(COCO_SPLITS),
                        help="COCO splits to search, in order (default: %(default)s)")
    parser.add_argument("--user-agent", default=os.environ.get("EGOGAP_USER_AGENT")
                        or DEFAULT_USER_AGENT,
                        help="HTTP User-Agent; Wikimedia requires a descriptive one")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="parallel COCO downloads (default: %(default)s)")
    parser.add_argument("--wiki-workers", type=int, default=DEFAULT_WIKI_WORKERS,
                        help="parallel Wikimedia downloads; keep it small, upload.wikimedia.org "
                             "throttles bulk clients (default: %(default)s)")
    parser.add_argument("--wiki-delay", type=float, default=DEFAULT_WIKI_DELAY,
                        help="seconds to pause before each Wikimedia download (default: %(default)s)")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                        help="retries per request (default: %(default)s)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="per-request timeout in seconds (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-download images that already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="only report what would happen")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    args.data_dir = args.data_dir.expanduser().resolve()
    if not args.data_dir.is_dir():
        LOG.error("data directory not found: %s", args.data_dir)
        return 2
    args.images_dir = (args.images_dir or args.data_dir / "images").expanduser().resolve()
    if args.coco_ann_dir is not None:
        args.coco_ann_dir = Path(args.coco_ann_dir).expanduser().resolve()

    steps: list[tuple[str, Callable[[argparse.Namespace], Stats]]] = []
    if args.target in ("all", "coco"):
        steps.append(("coco", cmd_coco))
    if args.target in ("all", "wikimedia"):
        steps.append(("wikimedia", cmd_wikimedia))
    if args.target in ("all", "fix-paths"):
        steps.append(("fix-paths", cmd_fix_paths))

    failed = 0
    for name, step in steps:
        LOG.info("=== %s ===", name)
        stats = step(args)
        failed += stats.failed
    if failed:
        LOG.error("finished with %d failure(s)", failed)
        return 1
    LOG.info("all done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
