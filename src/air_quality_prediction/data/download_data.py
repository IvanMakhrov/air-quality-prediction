import logging
from pathlib import Path
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

YANDEX_DISK_PUBLIC_KEY = "https://disk.yandex.ru/d/wMVfwD9hLs2Apw"
OUTPUT_DIR = Path("data/raw")
CSV_FILENAME = "air_weather_data_lite.csv"


def get_direct_download_url(public_url: str) -> str:
    """
    Get direct link for downloading file from public link Yandex.Disk
    from public API.
    """
    path = urlparse(public_url).path
    if "/d/" not in path:
        raise ValueError(
            "Invalid Yandex.Disk public link. Expected format: https://disk.yandex.ru/d/<key>"
        )
    public_key = path.split("/d/")[1].rstrip("/")
    logger.debug(f"Extracted public key: {public_key}")

    api_url = f"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=https://disk.yandex.ru/d/{public_key}"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/112.0 Safari/537.36"
    }

    logger.info("Requesting direct download URL from Yandex.Disk API...")
    response = requests.get(api_url, headers=headers, timeout=10)
    response.raise_for_status()

    data = response.json()
    download_url = data.get("href")
    if not download_url:
        raise RuntimeError(f"Failed to get download URL. API response: {data}")

    logger.info("Direct download URL obtained.")
    return download_url


def download_from_yandex_disk(public_key: str, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    try:
        direct_url = get_direct_download_url(public_key)
        logger.info("Downloading file...")
        response = requests.get(direct_url, stream=True, timeout=60)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Data saved to {output_path}")
        return output_path

    except requests.RequestException as e:
        logger.error(f"Download failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


def download_data() -> Path:
    return download_from_yandex_disk(
        public_key=YANDEX_DISK_PUBLIC_KEY,
        output_dir=OUTPUT_DIR,
        filename=CSV_FILENAME,
    )


if __name__ == "__main__":
    download_data()
