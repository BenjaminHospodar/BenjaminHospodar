#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import re
from functools import lru_cache
from pathlib import Path

from skyfield.api import Loader, Star, wgs84
from skyfield.data import hipparcos

DEFAULT_WIDTH = 120
DEFAULT_HEIGHT = 15
STAR_MAGNITUDE_CUTOFF = 6.5
HORIZON_CUTOFF_DEGREES = 0.5
BRIGHT_STAR_HALO_RADIUS = 1
FAINT_STAR_DENSITY_CUTOFF = 2.7
SPARSE_CELL_WIDTH = 12
SPARSE_CELL_HEIGHT = 3
REPO_ROOT = Path(__file__).resolve().parents[1]
SKYFIELD_CACHE_DIR = REPO_ROOT / ".skyfield-cache"
STAR_PRIORITY = {" ": 0, ".": 1, "+": 2, "*": 3}

START_MARKER = "<!-- NIGHT-SKY:START -->"
END_MARKER = "<!-- NIGHT-SKY:END -->"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_loader() -> Loader:
    return Loader(str(SKYFIELD_CACHE_DIR))


@lru_cache(maxsize=1)
def load_ephemeris() -> object:
    return get_loader()("de421.bsp")


@lru_cache(maxsize=1)
def load_timescale() -> object:
    return get_loader().timescale()


@lru_cache(maxsize=1)
def load_star_catalog() -> tuple[object, list[float]]:
    with get_loader().open(hipparcos.URL) as handle:
        dataframe = hipparcos.load_dataframe(handle)

    dataframe = dataframe[dataframe["ra_degrees"].notnull() & dataframe["dec_degrees"].notnull()]
    dataframe = dataframe[dataframe["magnitude"] <= STAR_MAGNITUDE_CUTOFF]

    stars = Star.from_dataframe(dataframe)
    magnitudes = dataframe["magnitude"].tolist()
    return stars, magnitudes


def build_observer(lat_deg: float, lon_deg: float):
    earth = load_ephemeris()["earth"]
    return earth + wgs84.latlon(lat_deg, lon_deg)


def iter_visible_star_points(
    observer,
    when_utc: dt.datetime,
) -> list[tuple[float, float, float]]:
    stars, magnitudes = load_star_catalog()
    time = load_timescale().from_datetime(when_utc)
    astrometric = observer.at(time).observe(stars)
    altitudes, azimuths, _distance = astrometric.apparent().altaz()

    points: list[tuple[float, float, float]] = []
    for azimuth_deg, altitude_deg, magnitude in zip(azimuths.degrees, altitudes.degrees, magnitudes):
        if altitude_deg <= 0.0:
            continue

        points.append((float(azimuth_deg) % 360.0, float(altitude_deg), float(magnitude)))

    return points


def star_char(magnitude: float) -> str:
    if magnitude <= 1.4:
        return "*"
    if magnitude <= 2.8:
        return "+"
    return "."


def star_seed(azimuth_deg: float, altitude_deg: float, magnitude: float) -> int:
    azimuth_key = int(round(azimuth_deg * 10.0))
    altitude_key = int(round(altitude_deg * 10.0))
    magnitude_key = int(round(magnitude * 100.0))
    return (
        azimuth_key * 73856093
        ^ altitude_key * 19349663
        ^ magnitude_key * 83492791
    ) & 0xFFFFFFFF


def generate_sky_lines(
    when_utc: dt.datetime,
    lat_deg: float,
    lon_deg: float,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> list[str]:
    grid = [[" " for _ in range(width)] for _ in range(height)]
    observer = build_observer(lat_deg, lon_deg)

    def put(x: int, y: int, ch: str, overwrite_dot: bool = True) -> None:
        if not (0 <= x < width and 0 <= y < height):
            return
        cur = grid[y][x]
        if cur == "*":
            return
        if ch == "*":
            grid[y][x] = "*"
            return
        if cur == " " or (overwrite_dot and cur == "."):
            grid[y][x] = ch

    def put_clamped(x: int, y: int, ch: str, overwrite_dot: bool = True) -> None:
        put(
            min(width - 1, max(0, x)),
            min(height - 1, max(0, y)),
            ch,
            overwrite_dot=overwrite_dot,
        )

    def project_star_to_grid(azimuth_deg: float, altitude_deg: float) -> tuple[int, int]:
        x = int((azimuth_deg / 360.0) * (width - 1))
        y = int(((90.0 - altitude_deg) / 90.0) * (height - 1))
        return min(width - 1, max(0, x)), min(height - 1, max(0, y))

    def place_star(x: int, y: int, magnitude: float) -> None:
        current = grid[y][x]
        incoming = star_char(magnitude)
        if STAR_PRIORITY[incoming] > STAR_PRIORITY[current]:
            put(x, y, incoming)

    def place_faint_dots(x: int, y: int, magnitude: float) -> None:
        if not 2.2 < magnitude <= 4.5:
            return

        put_clamped(x, y, ".", overwrite_dot=False)
        if magnitude <= 3.6:
            put_clamped(x + 1, y, ".", overwrite_dot=False)

    def place_halo(x: int, y: int, magnitude: float) -> None:
        if magnitude > 1.4:
            return

        for dy in range(-BRIGHT_STAR_HALO_RADIUS, BRIGHT_STAR_HALO_RADIUS + 1):
            for dx in range(-BRIGHT_STAR_HALO_RADIUS, BRIGHT_STAR_HALO_RADIUS + 1):
                distance = abs(dx) + abs(dy)
                if distance == 0:
                    continue
                if distance <= BRIGHT_STAR_HALO_RADIUS:
                    put_clamped(x + dx, y + dy, ".", overwrite_dot=False)

    visible_points = [
        (azimuth_deg, altitude_deg, magnitude)
        for azimuth_deg, altitude_deg, magnitude in iter_visible_star_points(observer, when_utc)
        if altitude_deg >= HORIZON_CUTOFF_DEGREES
    ]
    visible_points.sort(key=lambda item: (item[2], item[1]), reverse=True)

    for azimuth_deg, altitude_deg, magnitude in visible_points:
        x, y = project_star_to_grid(azimuth_deg, altitude_deg)
        if magnitude >= FAINT_STAR_DENSITY_CUTOFF:
            cell_x = x // SPARSE_CELL_WIDTH
            cell_y = y // SPARSE_CELL_HEIGHT
            seed = star_seed(azimuth_deg, altitude_deg, magnitude)
            cell_seed = (cell_x * 92821) ^ (cell_y * 68917)
            sparse_gate = (seed ^ cell_seed) % 100
            skip_threshold = 62 + int((magnitude - FAINT_STAR_DENSITY_CUTOFF) * 10)
            if sparse_gate < skip_threshold:
                continue

        place_star(x, y, magnitude)
        place_halo(x, y, magnitude)
        place_faint_dots(x, y, magnitude)

    return ["".join(row) for row in grid]


def build_block(lines: list[str]) -> str:
    normalized_lines = [line.rstrip()[:DEFAULT_WIDTH] for line in lines]
    if len(normalized_lines) > DEFAULT_HEIGHT:
        normalized_lines = normalized_lines[:DEFAULT_HEIGHT]
    elif len(normalized_lines) < DEFAULT_HEIGHT:
        normalized_lines.extend([""] * (DEFAULT_HEIGHT - len(normalized_lines)))

    while normalized_lines and not normalized_lines[0]:
        normalized_lines.pop(0)
    while normalized_lines and not normalized_lines[-1]:
        normalized_lines.pop()

    code = "\n".join(normalized_lines)
    return f"```text\n{code}\n```"


def update_readme_section(path: Path, block: str) -> bool:
    """Replace the night-sky section in *path*; appends the section if markers are absent."""
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    section = f"{START_MARKER}\n{block}\n{END_MARKER}"

    if START_MARKER in original and END_MARKER in original:
        pattern = re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER)
        updated = re.sub(pattern, section, original, flags=re.DOTALL)
    else:
        sep = "\n\n" if original.rstrip() else ""
        updated = original.rstrip() + sep + section + "\n"

    if updated == original:
        return False

    path.write_text(updated, encoding="utf-8")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a live ASCII night sky and embed it into README.md.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lat", type=float, metavar="DEGREES",
                        help="Observer latitude in decimal degrees (or set OBSERVER_LAT env var)")
    parser.add_argument("--lon", type=float, metavar="DEGREES",
                        help="Observer longitude in decimal degrees (or set OBSERVER_LON env var)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        lat = args.lat if args.lat is not None else float(os.environ["OBSERVER_LAT"])
        lon = args.lon if args.lon is not None else float(os.environ["OBSERVER_LON"])
    except KeyError as exc:
        _log.error("Observer coordinates required: use --lat/--lon or set %s env var.", exc.args[0])
        return 1
    except ValueError as exc:
        _log.error("Invalid coordinate value: %s", exc)
        return 1

    now_utc = dt.datetime.now(dt.timezone.utc)
    _log.info("Generating sky at %s UTC", now_utc.strftime("%Y-%m-%d %H:%M"))

    try:
        lines = generate_sky_lines(now_utc, lat, lon)
    except Exception as exc:
        _log.error("Failed to generate sky lines: %s", exc)
        return 1

    block = build_block(lines)
    readme_path = REPO_ROOT / "README.md"
    changed = update_readme_section(readme_path, block)
    _log.info("README %s", "updated" if changed else "unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())