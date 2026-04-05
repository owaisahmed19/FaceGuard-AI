import os
import requests
from config.settings import MODELS_DIR
from loguru import logger

def download_file(url, filepath):
    if not os.path.exists(filepath):
        logger.info(f"Downloading {url} to {filepath}...")
        r = requests.get(url, stream=True)
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info("Download complete.")
    else:
        logger.info(f"{filepath} already exists.")

detect_model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
recognize_model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

detect_path = os.path.join(MODELS_DIR, "face_detection_yunet.onnx")
recognize_path = os.path.join(MODELS_DIR, "face_recognition_sface.onnx")

if __name__ == "__main__":
    download_file(detect_model_url, detect_path)
    download_file(recognize_model_url, recognize_path)
