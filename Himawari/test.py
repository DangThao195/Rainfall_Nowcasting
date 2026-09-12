from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import bz2
import os
import requests

BASE_URL = "https://noaa-himawari9.s3.amazonaws.com"
BANDS = ["B08", "B09", "B10", "B13", "B14", "B15"]
SEGMENTS = ["03", "04", "05"]  # Bao phủ bbox Việt Nam

RAW_DIR = Path("data/himawari/raw")
DAT_DIR = Path("data/himawari/dat")
RAW_DIR.mkdir(parents=True, exist_ok=True)
DAT_DIR.mkdir(parents=True, exist_ok=True)


def latest_available_slot():
    # Chờ 20 phút để dữ liệu được ghi đầy đủ lên bucket
    time = datetime.now(timezone.utc) - timedelta(minutes=20)
    minute = 30 if time.minute >= 30 else 0
    return time.replace(minute=minute, second=0, microsecond=0)


def make_keys(slot):
    date = slot.strftime("%Y%m%d")
    hhmm = slot.strftime("%H%M")
    prefix = slot.strftime("AHI-L1b-FLDK/%Y/%m/%d/%H%M")

    return [
        (
            f"{prefix}/"
            f"HS_H09_{date}_{hhmm}_{band}_FLDK_R20_S{segment}10.DAT.bz2"
        )
        for band in BANDS
        for segment in SEGMENTS
    ]


def download(key):
    destination = RAW_DIR / Path(key).name
    temporary = Path(str(destination) + ".part")

    if destination.exists() and destination.stat().st_size > 0:
        return destination

    url = f"{BASE_URL}/{key}"

    with requests.get(url, stream=True, timeout=(10, 180)) as response:
        response.raise_for_status()

        with temporary.open("wb") as file:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    file.write(chunk)

    os.replace(temporary, destination)
    return destination


def decompress(source):
    destination = DAT_DIR / source.name.removesuffix(".bz2")
    temporary = Path(str(destination) + ".part")

    if destination.exists() and destination.stat().st_size > 0:
        return destination

    with bz2.open(source, "rb") as input_file:
        with temporary.open("wb") as output_file:
            while chunk := input_file.read(1024 * 1024):
                output_file.write(chunk)

    os.replace(temporary, destination)
    return destination


slot = latest_available_slot()
keys = make_keys(slot)

print("Observation slot:", slot.isoformat())
print("Expected files:", len(keys))  # 6 band × 3 segment = 18

with ThreadPoolExecutor(max_workers=4) as executor:
    compressed_files = list(executor.map(download, keys))

dat_files = [decompress(path) for path in compressed_files]

print(f"Downloaded and decompressed {len(dat_files)} files")

# Colab:
# !pip install -q satpy pyresample pyproj netcdf4

from satpy import Scene

scene = Scene(
    filenames=[str(path) for path in dat_files],
    reader="ahi_hsd",
)

scene.load(
    ["B08", "B09", "B10", "B13", "B14", "B15"],
    calibration="brightness_temperature",
)

print(scene["B13"])