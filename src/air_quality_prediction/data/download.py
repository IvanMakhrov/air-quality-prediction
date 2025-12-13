import requests
import zipfile
from pathlib import Path
from io import BytesIO
import logging

logger = logging.getLogger(__name__)


YANDEX_DISK_PUBLIC_KEY = "https://disk.360.yandex.ru/d/fjb72KvKlig9Mg"
OUTPUT_DIR = Path("data/raw")
CSV_FILENAME = "air_weather_data_lite.csv"

def download_from_yandex_disk(public_key: str, output_dir: Path, filename: str) -> Path:
    """
    Downloads a file from Yandex.Disk public link and extracts CSV if needed.
    
    Args:
        public_key: Public share link (e.g. https://disk.yandex.ru/d/AbC123)
        output_dir: Directory to save the file
        filename: Expected CSV filename (e.g. 'data.csv')

    Returns:
        Path to downloaded CSV file
    """
    logger.info(f"Downloading data from Yandex.Disk...")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    if "/d/" in public_key:
        direct_url = public_key.replace("/d/", "/download/")
    else:
        raise ValueError("Invalid Yandex.Disk public link format. Use /d/...")

    try:
        response = requests.get(direct_url, stream=True, timeout=30)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Data saved to {output_path}")
        return output_path

    except requests.RequestException as e:
        logger.error(f"Download failed: {e}")
        raise

def download_data() -> Path:
    """
    Public API function for DVC pipeline integration.
    """
    return download_from_yandex_disk(
        public_key=YANDEX_DISK_PUBLIC_KEY,
        output_dir=OUTPUT_DIR,
        filename=CSV_FILENAME
    )
