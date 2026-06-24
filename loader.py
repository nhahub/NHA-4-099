import logging
import os
import tempfile
import requests

logger = logging.getLogger("loader")


def download_file(url: str) -> bytes:
    logger.info("Downloading — url=%s", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    content = response.content
    logger.info("Downloaded %d bytes from %s", len(content), url)
    return content