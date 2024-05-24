from pathlib import Path
import shutil

src = Path('./outputs')

image_list = list(src.glob('*.png'))
sub_folder_names = set(["_".join(image.stem.split('_')[:2]) for image in image_list])
for sub_folder_name in sub_folder_names:
    tgt_folder = src / sub_folder_name
    tgt_folder.mkdir(parents=True, exist_ok=True)
    for image in image_list:
        if sub_folder_name in image.stem:
            shutil.copy(image, tgt_folder / image.name)
