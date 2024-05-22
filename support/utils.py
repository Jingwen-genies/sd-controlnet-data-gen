import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import shutil
import csv
import pandas as pd


def read_json(file_path):
    try:
        with open(file_path, "r", encoding='utf-8') as file:
            return json.load(file)
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def write_json(data, file_path):
    try:
        with open(file_path, "w", encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except IOError as e:
        print(f"Error writing file {file_path}: {e}")


def generate_csv(input_folder, csv_file, overwrite=False, type: str = "subfolder"):
    """
    Generate a csv file that contains the file names of the images and the corresponding json file.
    Args:
        input_folder: the folder that contains the images and the json files
        type: ["subfolder", "same_folder"]
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

    if type == "subfolder":
        # Handle images in subfolders
        print("generating csv for images in subfolders")
        for sub_folder in input_folder.iterdir():
            if sub_folder.is_dir():
                initial_landmark_path = sub_folder / "pose.json"
                for file in sub_folder.iterdir():
                    if file.suffix in [".jpg", ".png", ".jpeg"] and "input" not in file.stem:
                        file_list.append(file.resolve())
                        initial_json_list.append(initial_landmark_path.resolve())

    elif type == "same_folder":
        print("generating csv for images in the same folder")
        # Handle images in the same folder
        initial_landmark_path = ""
        for file in input_folder.iterdir():
            print(file)
            if file.is_file() and file.suffix in [".jpg", ".png", ".jpeg"]:
                file_list.append(file.resolve())
                initial_json_list.append(initial_landmark_path)

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image", "new_landmark", "is_kept"])
        for i in range(len(file_list)):
            print(f"Processing {file_list[i]}")
            # write the row, divide the columns by comma
            sub_folder = file_list[i].parent
            new_landmark_name = sub_folder/(file_list[i].stem + "_landmark.json")
            if initial_json_list[i] != "" and initial_json_list[i].exists():
                shutil.copy(initial_json_list[i], new_landmark_name)
            print([file_list[i], new_landmark_name.resolve(), True])

            writer.writerow([file_list[i], new_landmark_name.resolve(), True])


def read_openpose(json_path):
    """
    Read the json file that contains the openpose keypoints.and possible bounding box
    Args:
        json_path: path to the json file

    Returns:
        a dictionary that contains the openpose keypoints
    """
    data = read_json(json_path)
    print(f"Reading openpose json file: {json_path}")
    if data:
        landmark = data["people"][0].get('face_keypoints_2d')
        if landmark:# list
            landmark = np.array(landmark).reshape(-1, 3)
            keypoints = landmark[:, :2]
            # for i in range(len(keypoints)):
            #     print(f"keypoint {i}: {i + 24} {landmark[i]}")
            # make sure the landmark is in the range of [0, 1], if not read the image and get the landmark
            if np.max(keypoints) > 1:
                print("The landmark is not in the range of [0, 1], please provide the image path to get the correct landmark")
                print(landmark)
                if "canvas_width" in data:
                    image_width = data["canvas_width"]
                    image_height = data["canvas_height"]
                else:
                    image_width, image_height = 512, 512
                landmark[:, 0] /= image_width
                landmark[:, 1] /= image_height
        bbox = data["people"][0].get("bbox")
        return landmark, bbox
    return None, None


def write_openpose(landmark, bbox, json_path, canvas_width = 512, canvas_height = 512):
    """
    Write the openpose keypoints to a json file
    Args:
        landmark: the keypoints
        bbox: the bounding box
        json_path: the path to the json file
    """
    data = {
        "people": [
            {
                "face_keypoints_2d": landmark.flatten().tolist(),
            }
        ],
        "canvas_width": canvas_width,
        "canvas_height": canvas_height
    }
    if bbox:
        data["people"][0]["bbox"] = bbox
    write_json(data, json_path)


def load_image_and_landmarks(image, landmarks_json):
    r = 5
    # img = Image.open(image).convert("RGBA")
    img = image
    original_img = img.copy()
    draw = ImageDraw.Draw(img)

    landmark, bbox = read_openpose(landmarks_json)
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


        

