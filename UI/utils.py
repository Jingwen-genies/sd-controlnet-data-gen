import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import shutil
import csv
import pandas as pd


def read_json(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def write_json(data, file_path):
    try:
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
    except IOError as e:
        print(f"Error writing file {file_path}: {e}")


def generate_csv(input_folder, csv_file, overwrite=False):
    """
    Generate a csv file that contains the file names of the images and the corresponding json file.
    Args:
        input_folder: the folder that contains the images and the json files
    Args:
        input_folder:

    Returns:
    """
    input_folder = Path(input_folder)
    if csv_file.exists() and not overwrite:
        print(f"The file {csv_file} already exists. If you want to overwrite it, please set the overwrite flag to True.")
        return
    # if the file exists, and overwrite is True, then remove the file
    if csv_file.exists():
        csv_file.unlink()
    file_list = []
    initial_json_list = []
    for sub_folder in input_folder.iterdir():
        if sub_folder.is_dir():
            initial_landmark_path = sub_folder / "pose.json"
            print(initial_landmark_path)
            for file in sub_folder.iterdir():
                if file.suffix in [".jpg", ".png", ".jpeg"] and "input" not in file.stem:
                    # get the absolute path of file
                    file_list.append(file.resolve())
                    initial_json_list.append(initial_landmark_path.resolve())
                    print(file.resolve(), initial_landmark_path.resolve())

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image", "new_landmark", "is_kept"])
        for i in range(len(file_list)):
            # write the row, divide the columns by comma
            sub_folder = file_list[i].parent
            new_landmark_name = sub_folder/(file_list[i].stem + "_landmark.json")
            shutil.copy(initial_json_list[i], new_landmark_name)
            writer.writerow([file_list[i], new_landmark_name.resolve(), True])


def read_openpose(json_path):
    """
    Read the json file that contains the openpose keypoints.
    Args:
        json_path: path to the json file

    Returns:
        a dictionary that contains the openpose keypoints
    """
    data = read_json(json_path)
    if data:
        landmark = data["people"][0]['face_keypoints_2d']  # list
        landmark = np.array(landmark).reshape(-1, 3)
        # make sure the landmark is in the range of [0, 1], if not read the image and get the landmark
        if np.max(landmark) > 1:
            image_width = data["canvas_width"]
            image_height = data["canvas_height"]
            landmark[:, 0] /= image_width
            landmark[:, 1] /= image_height
        return landmark
    return None


def load_image_and_landmarks(image, landmarks_json):
    r = 5
    # img = Image.open(image).convert("RGBA")
    img = image
    original_img = img.copy()
    draw = ImageDraw.Draw(img)

    landmark = read_openpose(landmarks_json)
    keypoint = []
    for pt in landmark:
        if 0 < pt[0] < 1 and 0 < pt[1] < 1:
            x = pt[0] * img.width
            y = pt[1] * img.height
        else:
            x = pt[0]
            y = pt[1]
        if len(pt) == 2:
            v = 2
        else:
            v = (2 if pt[-1] is False else 1)
        keypoint.append((x, y, v))
    for pt in keypoint:
        x, y, v = pt
        draw.ellipse((x - r, y - r, x + r, y + r), fill='yellow', outline='yellow')
    return image


if __name__ == "__main__":
    input_folder = "../outputs"
    csv_file = Path(input_folder) / "synthetic_data_info.csv"
    generate_csv(input_folder, csv_file, overwrite=True)


        

