import os
from pathlib import Path
from PIL import Image

# 数据集
data_dir = Path('data')


valid_formats = {'JPEG', 'PNG'}


for img_path in data_dir.glob("**/*"):

    if img_path.is_file():
        with Image.open(img_path) as img:
            actual_format = img.format

            if actual_format not in valid_formats:
                print(f"Deleting {img_path}, format is {actual_format}")
                os.remove(img_path)

