"""Download the public validation subsets used by the research notebooks.

The downloaded files live under data/external, which is ignored by Git.  The
script intentionally fetches only the subsets needed by the reproducible
checks, not the complete multi-gigabyte datasets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
import struct
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DATA = ROOT / "data" / "external"

EYEDENTIFY_DATASET = "vijuls/pupildiameterdatasets"
EYEDENTIFY_API = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    f"{EYEDENTIFY_DATASET}"
)
EYEDENTIFY_METADATA_API = (
    "https://www.kaggle.com/api/v1/datasets/view/"
    f"{EYEDENTIFY_DATASET}"
)

LPW_FILES = {
    "README.txt": (
        "https://darus.uni-stuttgart.de/api/access/datafile/198470",
        None,
    ),
    "subject_01/9.txt": (
        "https://darus.uni-stuttgart.de/api/access/datafile/198421",
        "ae542d07cbb0c4e191e7893962eaeec6",
    ),
    "subject_01/9.avi": (
        "https://darus.uni-stuttgart.de/api/access/datafile/198424",
        "f92e675ba0e21bcb83a009cb82291a8e",
    ),
}

LPW_CENTER_FILES = {
    "1/9.avi": (198424, "f92e675ba0e21bcb83a009cb82291a8e"),
    "1/9.txt": (198421, "ae542d07cbb0c4e191e7893962eaeec6"),
    "2/10.avi": (198530, "ed916fd9d1d2618ab347b45f07527c7a"),
    "2/10.txt": (198519, "ea4ec26bc39ff878e1e8a288aca6a94f"),
    "3/16.avi": (198437, "5ae38af011ec41749a6baab3c6790fba"),
    "3/16.txt": (198453, "a6edd842dac96e03db5c26a3612a747a"),
    "4/2.avi": (198428, "aca1acf87d7d46721280b8108955bca1"),
    "4/2.txt": (198405, "1fcd0c8f48cc671e62c99c10e9ea015d"),
    "5/10.avi": (198481, "b11669fcb730484a7fe29641ec9a1d49"),
    "5/10.txt": (198457, "40923b0a1f46b9658f95d9f926f9d0b9"),
    "6/13.avi": (198471, "7f90ef9978aeb2d235f2d4979f99bfe7"),
    "6/13.txt": (198412, "1e60b34f06764cbb216ad9ab790418ff"),
    "7/21.avi": (198506, "4a5255986b63c95362d60d95d1aca578"),
    "7/21.txt": (198477, "8d587d0d6a2f8306eea00b2adb5e3ee3"),
    "8/9.avi": (198439, "cb781f0360140915fce51f267a5e1871"),
    "8/9.txt": (198419, "aab289e199e62cfd24fabbbd2b525e32"),
}

SWIRSKI_ARCHIVES = {
    "p1-left": (
        "https://www.cl.cam.ac.uk/research/rainbow/projects/"
        "pupiltracking/files/p1-left.zip"
    ),
    "p2-left": (
        "https://www.cl.cam.ac.uk/research/rainbow/projects/"
        "pupiltracking/files/p2-left.zip"
    ),
}

NEMAR_MANIFEST_URL = (
    "https://data.nemar.org/nm000150/v1.0.0/manifest.json"
)
NEMAR_PATTERN = re.compile(
    r"^derivatives/sub-\d+/ses-0[12]/eyetrack/"
    r"sub-\d+_ses-0[12]_task-stim01_desc-"
    r"(pupil_eyetrack|gaze_visualangle_eyetrack)\.tsv\.gz$"
)
NEMAR_METADATA = {
    "README.md",
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
}


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download(url: str, destination: Path, checksum: str | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    algorithm = "md5" if checksum and len(checksum) == 32 else "sha256"
    if destination.exists() and (
        checksum is None or digest(destination, algorithm) == checksum
    ):
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            shutil.copyfileobj(response.raw, stream)
    temporary.replace(destination)

    if checksum is not None and digest(destination, algorithm) != checksum:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch: {destination}")
    return destination


def fetch_lpw() -> None:
    target = EXTERNAL_DATA / "lpw_sample"
    for relative_path, (url, checksum) in LPW_FILES.items():
        path = download(url, target / relative_path, checksum)
        print(f"LPW: {path.relative_to(ROOT)}")


def fetch_lpw_center() -> None:
    """Fetch one annotated LPW video from each of eight participants."""
    target = EXTERNAL_DATA / "lpw_center"

    def fetch(item: tuple[str, tuple[int, str]]) -> Path:
        relative_path, (file_id, checksum) = item
        return download(
            f"https://darus.uni-stuttgart.de/api/access/datafile/{file_id}",
            target / relative_path,
            checksum,
        )

    with ThreadPoolExecutor(max_workers=6) as executor:
        paths = list(executor.map(fetch, LPW_CENTER_FILES.items()))
    for path in paths:
        print(f"LPW-center: {path.relative_to(ROOT)}")
    (target / "subset_manifest.json").write_text(
        json.dumps(
            {
                "dataset": "Labelled Pupils in the Wild (LPW)",
                "license": "CC BY-NC-SA 4.0",
                "selection": "one video from each of participants 1-8",
                "training_participants": [1, 2, 3, 4, 5, 6],
                "validation_participants": [7],
                "test_participants": [8],
                "files": [str(path.relative_to(target)).replace("\\", "/") for path in paths],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def fetch_swirski() -> None:
    target = EXTERNAL_DATA / "swirski"
    for name, url in SWIRSKI_ARCHIVES.items():
        archive = download(url, target / f"{name}.zip")
        extracted = target / name
        labels = extracted / "pupil-ellipses.txt"
        if not labels.exists():
            extracted.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(extracted)
        print(f"Swirski: {labels.relative_to(ROOT)}")


def fetch_nemar() -> None:
    manifest = requests.get(NEMAR_MANIFEST_URL, timeout=60).json()
    selected = [
        entry
        for entry in manifest
        if NEMAR_PATTERN.match(entry["path"])
        or entry["path"] in NEMAR_METADATA
    ]
    target = EXTERNAL_DATA / "nm000150_subset"

    def fetch(entry: dict) -> dict:
        checksum = (
            entry["checksum"]
            if entry.get("checksum_algorithm") == "sha256"
            else None
        )
        path = download(entry["bytes_url"], target / entry["path"], checksum)
        print(f"NEMAR: {path.relative_to(ROOT)}")
        return {
            "path": entry["path"],
            "size": entry["size"],
            "checksum_algorithm": entry["checksum_algorithm"],
            "checksum": entry["checksum"],
            "bytes_url": entry["bytes_url"],
        }

    with ThreadPoolExecutor(max_workers=8) as executor:
        downloaded = list(executor.map(fetch, selected))

    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "subset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": "NEMAR nm000150 v1.0.0",
                "source_manifest": NEMAR_MANIFEST_URL,
                "selection": "stim01, sessions 01/02, pupil and gaze derivatives",
                "files": downloaded,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"NEMAR manifest: {manifest_path.relative_to(ROOT)}")


def _http_range(url: str, start: int, end: int) -> bytes:
    for attempt in range(5):
        try:
            response = requests.get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=(30, 180),
            )
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError("Server ignored HTTP Range")
            return response.content
        except (requests.RequestException, RuntimeError):
            if attempt == 4:
                raise
            time.sleep(0.5 * (2**attempt))
    raise RuntimeError("Unreachable")


def _eyedentify_archive_url() -> str:
    response = requests.get(EYEDENTIFY_API, allow_redirects=False, timeout=60)
    response.raise_for_status()
    location = response.headers.get("location")
    if not location:
        raise RuntimeError("Kaggle did not return a signed archive URL")
    return location


def _zip64_central_directory(url: str) -> tuple[int, int]:
    tail = requests.get(url, headers={"Range": "bytes=-65536"}, timeout=90)
    tail.raise_for_status()
    match = re.search(r"/(\d+)$", tail.headers.get("Content-Range", ""))
    if not match:
        raise RuntimeError("Missing archive size in Content-Range")
    total_size = int(match.group(1))
    locator_index = tail.content.rfind(bytes.fromhex("504b0607"))
    if locator_index < 0:
        raise RuntimeError("ZIP64 locator not found")
    _, _, zip64_offset, _ = struct.unpack_from(
        "<4sLQL", tail.content, locator_index
    )
    record = _http_range(url, zip64_offset, zip64_offset + 55)
    values = struct.unpack("<4sQ2H2L4Q", record)
    central_size = int(values[-2])
    central_offset = int(values[-1])
    if central_offset + central_size > total_size:
        raise RuntimeError("Invalid ZIP64 central directory bounds")
    return central_offset, central_size


def _zip64_values(
    extra: bytes,
    compressed_size: int,
    uncompressed_size: int,
    local_offset: int,
) -> tuple[int, int, int]:
    cursor = 0
    while cursor + 4 <= len(extra):
        field_id, field_size = struct.unpack_from("<HH", extra, cursor)
        field = extra[cursor + 4 : cursor + 4 + field_size]
        cursor += 4 + field_size
        if field_id != 0x0001:
            continue
        field_cursor = 0
        if uncompressed_size == 0xFFFFFFFF:
            uncompressed_size = struct.unpack_from("<Q", field, field_cursor)[0]
            field_cursor += 8
        if compressed_size == 0xFFFFFFFF:
            compressed_size = struct.unpack_from("<Q", field, field_cursor)[0]
            field_cursor += 8
        if local_offset == 0xFFFFFFFF:
            local_offset = struct.unpack_from("<Q", field, field_cursor)[0]
        break
    return int(compressed_size), int(uncompressed_size), int(local_offset)


def _select_remote_members(
    url: str,
    participants: tuple[int, ...],
    frame_numbers: tuple[int, ...],
) -> list[dict]:
    central_offset, central_size = _zip64_central_directory(url)
    prefix = "eyedentify/eyedentify/left_eyes/"
    participants_set = set(participants)
    frame_names = {f"frame_{number:02d}.png" for number in frame_numbers}
    selected: list[dict] = []
    position = central_offset
    remaining = central_size
    buffer = b""
    started_left_eyes = False
    chunk_size = 8 * 1024 * 1024
    central_struct = struct.Struct("<4s6H3L5H2L")

    while remaining > 0:
        request_size = min(chunk_size, remaining)
        chunk = _http_range(url, position, position + request_size - 1)
        position += len(chunk)
        remaining -= len(chunk)
        buffer += chunk
        cursor = 0
        while len(buffer) - cursor >= central_struct.size:
            values = central_struct.unpack_from(buffer, cursor)
            if values[0] != bytes.fromhex("504b0102"):
                remaining = 0
                break
            flags = values[3]
            compression = values[4]
            crc = values[7]
            compressed_size = values[8]
            uncompressed_size = values[9]
            filename_size = values[10]
            extra_size = values[11]
            comment_size = values[12]
            local_offset = values[16]
            record_size = central_struct.size + filename_size + extra_size + comment_size
            if len(buffer) - cursor < record_size:
                break
            filename_bytes = buffer[
                cursor + central_struct.size : cursor + central_struct.size + filename_size
            ]
            extra_start = cursor + central_struct.size + filename_size
            extra = buffer[extra_start : extra_start + extra_size]
            encoding = "utf-8" if flags & 0x800 else "cp437"
            name = filename_bytes.decode(encoding)
            compressed_size, uncompressed_size, local_offset = _zip64_values(
                extra, compressed_size, uncompressed_size, local_offset
            )
            cursor += record_size

            if name.startswith(prefix):
                started_left_eyes = True
                parts = name[len(prefix) :].split("/")
                if len(parts) == 3 and parts[0].isdigit():
                    participant = int(parts[0])
                    if participant in participants_set and (
                        parts[2] == "session_data.csv" or parts[2] in frame_names
                    ):
                        selected.append(
                            {
                                "name": name,
                                "filename_size": filename_size,
                                "compression": compression,
                                "crc": crc,
                                "compressed_size": compressed_size,
                                "uncompressed_size": uncompressed_size,
                                "local_offset": local_offset,
                            }
                        )
            elif started_left_eyes:
                remaining = 0
                break
        buffer = buffer[cursor:]
        print(
            f"\rEyeDentify: índice remoto {len(selected):,} archivos seleccionados",
            end="",
            flush=True,
        )
    print()
    return selected


def _extract_remote_member(url: str, member: dict, target: Path) -> Path:
    if target.exists() and target.stat().st_size == member["uncompressed_size"]:
        return target
    start = member["local_offset"]
    anticipated = 30 + member["filename_size"] + 4096 + member["compressed_size"]
    payload = _http_range(url, start, start + anticipated - 1)
    local = struct.unpack_from("<4s5H3L2H", payload)
    if local[0] != bytes.fromhex("504b0304"):
        raise RuntimeError(f"Invalid local ZIP header for {member['name']}")
    filename_size, extra_size = local[-2:]
    data_start = 30 + filename_size + extra_size
    data_end = data_start + member["compressed_size"]
    if data_end > len(payload):
        payload = _http_range(url, start, start + data_end - 1)
    compressed = payload[data_start:data_end]
    if member["compression"] == 0:
        content = compressed
    elif member["compression"] == 8:
        content = zlib.decompress(compressed, -15)
    else:
        raise RuntimeError(
            f"Unsupported compression {member['compression']} for {member['name']}"
        )
    if len(content) != member["uncompressed_size"]:
        raise RuntimeError(f"Size mismatch for {member['name']}")
    if zlib.crc32(content) & 0xFFFFFFFF != member["crc"]:
        raise RuntimeError(f"CRC mismatch for {member['name']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.write_bytes(content)
    temporary.replace(target)
    return target


def _allocate_stratified(total: int, sizes: list[int]) -> list[int]:
    raw = [size * total / sum(sizes) for size in sizes]
    allocated = [math.floor(value) for value in raw]
    for index in sorted(
        range(len(sizes)), key=lambda item: raw[item] - allocated[item], reverse=True
    )[: total - sum(allocated)]:
        allocated[index] += 1
    return allocated


def _participant_stratified_splits(rows: list[dict]) -> dict[int, str]:
    """Split people while balancing their target-diameter distributions."""
    values: dict[int, list[float]] = {}
    for row in rows:
        values.setdefault(row["participant_id"], []).append(
            float(row["left_pupil_mm"])
        )
    ordered = sorted(values, key=lambda participant: np_mean(values[participant]))
    stratum_count = min(4, len(ordered))
    strata = [
        ordered[round(index * len(ordered) / stratum_count) : round((index + 1) * len(ordered) / stratum_count)]
        for index in range(stratum_count)
    ]
    participant_count = len(ordered)
    validation_total = round(participant_count * 0.15)
    test_total = participant_count - round(participant_count * 0.70) - validation_total
    sizes = [len(stratum) for stratum in strata]
    validation_counts = _allocate_stratified(validation_total, sizes)
    test_counts = _allocate_stratified(test_total, sizes)

    rng = random.Random(42)
    split_by_participant: dict[int, str] = {}
    for stratum, validation_count, test_count in zip(
        strata, validation_counts, test_counts
    ):
        rng.shuffle(stratum)
        for participant in stratum[:test_count]:
            split_by_participant[participant] = "test"
        for participant in stratum[test_count : test_count + validation_count]:
            split_by_participant[participant] = "validation"
        for participant in stratum[test_count + validation_count :]:
            split_by_participant[participant] = "train"
    return split_by_participant


def np_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def fetch_eyedentify(
    participants: tuple[int, ...] = tuple(range(1, 21)),
    frames_per_session: int = 2,
) -> None:
    """Fetch a compact, participant-disjoint EyeDentify research subset.

    EyeDentify contains very redundant 30 FPS bursts.  We keep evenly spaced
    frames from every session while retaining many people and all lighting
    conditions.  The full dataset is about 34 GB; this subset is intentionally
    small enough to reproduce on a CPU workstation.
    """
    if frames_per_session < 1:
        raise ValueError("frames_per_session must be at least 1")
    target = EXTERNAL_DATA / "eyedentify"
    target.mkdir(parents=True, exist_ok=True)

    metadata = requests.get(EYEDENTIFY_METADATA_API, timeout=60)
    metadata.raise_for_status()
    (target / "dataset_metadata.json").write_text(
        json.dumps(metadata.json(), indent=2), encoding="utf-8"
    )
    (target / "LICENSE_NOTICE.md").write_text(
        "# EyeDentify local subset\n\n"
        "Source: https://www.kaggle.com/datasets/vijuls/pupildiameterdatasets\n\n"
        "License: CC BY-NC 4.0. This local copy is for research and "
        "non-commercial use. Cite Shah et al., *EyeDentify: A Dataset for "
        "Pupil Diameter Estimation Based on Webcam Images* (2024).\n",
        encoding="utf-8",
    )

    frame_numbers = tuple(
        round(15 + index * 60 / max(1, frames_per_session - 1))
        for index in range(frames_per_session)
    )
    archive_url = _eyedentify_archive_url()
    members = _select_remote_members(archive_url, participants, frame_numbers)
    if not members:
        raise RuntimeError("No EyeDentify members matched the requested subset")

    prefix = "eyedentify/eyedentify/"

    def extract(member: dict) -> Path:
        local = target / member["name"][len(prefix) :]
        return _extract_remote_member(archive_url, member, local)

    with ThreadPoolExecutor(max_workers=24) as executor:
        extracted = list(executor.map(extract, members))
    available_csvs = [path for path in extracted if path.name == "session_data.csv"]

    selected_rows: list[dict] = []
    selected_frame_names = {f"frame_{number:02d}.png" for number in frame_numbers}
    for csv_path in sorted(available_csvs):
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            relative_frame = Path(row.get("frame_path", "").replace("\\", "/"))
            if relative_frame.name not in selected_frame_names:
                continue
            image_path = target / "left_eyes" / relative_frame
            if not image_path.exists() or not row.get("left_pupil"):
                continue
            participant = int(relative_frame.parts[0])
            session = int(relative_frame.parts[1])
            selected_rows.append(
                {
                    "participant_id": participant,
                    "session_id": session,
                    "timestamp": row.get("#timestamp", ""),
                    "left_pupil_mm": row["left_pupil"],
                    "right_pupil_mm": row.get("right_pupil", ""),
                    "pupil_diameter_mm": row.get("pupil_diameter", ""),
                    "gaze_x": row.get("gaze_x", ""),
                    "gaze_y": row.get("gaze_y", ""),
                    "image_path": image_path.relative_to(target).as_posix(),
                }
            )

    participant_ids = sorted({row["participant_id"] for row in selected_rows})
    split_by_participant = _participant_stratified_splits(selected_rows)
    for row in selected_rows:
        row["split"] = split_by_participant[row["participant_id"]]

    fieldnames = [
        "participant_id",
        "session_id",
        "timestamp",
        "left_pupil_mm",
        "right_pupil_mm",
        "pupil_diameter_mm",
        "gaze_x",
        "gaze_y",
        "image_path",
        "split",
    ]
    manifest_path = target / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    split_summary = {
        split: {
            "participants": sorted(
                participant
                for participant, assigned in split_by_participant.items()
                if assigned == split
            ),
            "frames": sum(row["split"] == split for row in selected_rows),
        }
        for split in ("train", "validation", "test")
    }
    subset_manifest = {
        "dataset": "EyeDentify",
        "kaggle_ref": EYEDENTIFY_DATASET,
        "license": "CC BY-NC 4.0",
        "sampling": (
            f"frames {list(frame_numbers)} from each 3-second session; "
            "all 50 lighting sessions retained"
        ),
        "requested_participants": list(participants),
        "available_participants": participant_ids,
        "available_sessions": len(available_csvs),
        "downloaded_frames": len(selected_rows),
        "participant_disjoint_splits": split_summary,
        "split_strategy": (
            "participant-disjoint, four strata by participant mean left-pupil "
            "diameter, deterministic seed 42"
        ),
    }
    (target / "subset_manifest.json").write_text(
        json.dumps(subset_manifest, indent=2), encoding="utf-8"
    )
    print(f"EyeDentify: {manifest_path.relative_to(ROOT)}")
    print(json.dumps(split_summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=("eyedentify", "lpw", "lpw-center", "swirski", "nemar"),
        default=("lpw", "swirski", "nemar"),
    )
    parser.add_argument(
        "--eyedentify-participants",
        default="1-20",
        help="Participant range, for example 1-20",
    )
    parser.add_argument("--frames-per-session", type=int, default=4)
    args = parser.parse_args()
    selected = args.datasets or ("lpw", "swirski", "nemar")
    bounds = [int(value) for value in args.eyedentify_participants.split("-")]
    participants = tuple(range(bounds[0], bounds[-1] + 1))
    actions = {
        "eyedentify": lambda: fetch_eyedentify(
            participants, args.frames_per_session
        ),
        "lpw": fetch_lpw,
        "lpw-center": fetch_lpw_center,
        "swirski": fetch_swirski,
        "nemar": fetch_nemar,
    }
    for name in selected:
        actions[name]()


if __name__ == "__main__":
    main()
