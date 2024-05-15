""" Generate video/gif for generated image in outputs folder """
from pathlib import Path


output_folder = Path("outputs_bigeyes2_batch3")
images = []
# iterate the folder
for subfolder in output_folder.iterdir():
    if subfolder.is_dir():
        for file in subfolder.iterdir():
            if file.suffix == ".png" and "input" not in file.stem:
                images.append(file.as_posix())

# generatei gif from images
n_images = len(images)
if n_images > 100:
    # skip every n images/100 images
    images = images[::n_images//100]

# generate gif
from PIL import Image
imgs = [Image.open(image) for image in images]
gif_path = output_folder / "output.gif"
imgs[0].save(gif_path.as_posix(), save_all=True, append_images=imgs[1:], duration=100, loop=0)
print(f"Generated gif: {gif_path}")