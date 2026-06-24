import logging
import os

import requests

from config import DATA_DIR

logger = logging.getLogger("loader")


def download_file(url: str, output_dir: str = DATA_DIR) -> str:
    
    os.makedirs(output_dir, exist_ok=True)

    filename = url.split("/")[-1].split("?")[0] or "downloaded_file"
    output_path = os.path.join(output_dir, filename)

    logger.info("Downloading — url=%s", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    logger.info("Saved — path=%s  size=%d bytes", output_path, os.path.getsize(output_path))
    return os.path.abspath(output_path)