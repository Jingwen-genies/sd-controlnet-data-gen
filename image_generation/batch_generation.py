import subprocess
import logging
import random
import os
from pathlib import Path
import io
import base64
from PIL import Image
import rootutils
import csv
import warnings
import shutil
import json
import requests
import time
import cv2
import numpy as np

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="pytorch_lightning.utilities.distributed",
    message=".*rank_zero_only has been deprecated.*",
)
project_root = rootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    dotenv=True,
    pythonpath=True,
    cwd=False,
)
os.chdir(project_root)

from avatar_generation import (
    subject,
    subject_list,
    age,
    style,
    headwear,
    eye_color,
    hair_color,
    hair_style,
    skin_color,
    skin_texture,
    distinguishing_marks,
)
from avatar_generation import ControlnetRequest
from avatar_generation.support.utils import read_openpose, write_openpose


def setup_logging(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a file handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)

    # Create a stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Create a logging format
    formatter = logging.Formatter(
        "%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the handlers to the logger
    # logger.addHandler(handler)
    logger.addHandler(stream_handler)
    return logger


logger = setup_logging("./avatar_generation/logs.log")


def start_services(ports):
    """
    Starts the webui service on the given ports. Don't need to start survice anymore as we have an API server running
    """
    processes = []
    for port in ports:
        cmd = f"webui --api --port {port} --skip-install --api-server-stop"
        try:
            process = subprocess.Popen(cmd, shell=True)
            processes.append(process)
            logger.info("Started webui service on port %s.", port)
            time.sleep(1)
        except Exception as e:
            logger.error("Failed to start service on port %s: %s", port, e)
    return processes


def read_image(image_path):
    img = cv2.imread(image_path)
    retval, bytes = cv2.imencode(".png", img)
    encoded_image = base64.b64encode(bytes).decode("utf-8")
    return encoded_image


def get_pose_estimation(input_image_path, save_path="./pose.json"):
    # logger.info("Generating controls from openpose.")
    logger.info("detecting pose from image %s", input_image_path)
    embedded_image = read_image(input_image_path)
    url = f"https://stablediffusion.dev.genies.com:7860/controlnet/detect"
    payload = {
        "controlnet_module": "openpose_faceonly",
        "controlnet_input_images": [embedded_image],
        "controlnet_processor_res": 512,
        "controlnet_threshold_a": 64,
        "controlnet_threshold_b": 64,
        "low_vram": False,
    }
    response = requests.post(url, json=payload)
    print(response, flush=True)
    output = response.json()
    if output["info"] != "Success":
        logger.error(
            "Failed to get pose estimation from image %s",
            input_image_path,
        )
        return response.json()
    logging.info(output)
    pose = output["poses"][0]
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(pose, f, ensure_ascii=False, indent=4)
    return response.json()


def generate_prompt(bald_rate=0.9):
    selected_object = random.choice(subject_list)
    selected_age = random.choice(age)
    selected_style = random.choice(style)
    selected_eye_color = random.choice(eye_color)
    selected_hair_color = random.choice(hair_color)
    selected_hair_style = random.choice(hair_style)
    selected_skin_color = random.choice(skin_color)
    selected_skin_texture = random.choice(skin_texture)
    selected_distinctive_marks = random.choice(distinguishing_marks)
    is_bald = random.random() > (1 - bald_rate)
    print("is_bald", is_bald)
    if is_bald and selected_object in subject["Human Characters"]:
        prompt = (
            f"A {selected_age} {selected_object} with {selected_style} style. The character should have "
            f"{selected_eye_color} eyes. The character should be [bald], with no hair at all. "
            f"The skin should be {selected_skin_color} and {selected_skin_texture}."
        )
    else:
        prompt = (
            f"A {selected_age} {selected_object} with {selected_style} style. The character should have "
            f"{selected_eye_color} eyes. The hair should be {selected_hair_color} and "
            f"{selected_hair_style}. The skin should be {selected_skin_color} and {selected_skin_texture}. "
        )
    with_distinguishing_marks = random.random() > 0.7
    if with_distinguishing_marks:
        prompt += f"The character should have {selected_distinctive_marks}."
    return prompt


def generate_unique_prompt(history, bald_rate=0.9, max_attempts=100):
    for _ in range(max_attempts):
        prompt = generate_prompt(bald_rate)
        if prompt not in history:
            return prompt
    return None


def get_bbox_from_contours(input_file_list):
    bboxes = []
    for input_file in input_file_list:
        input_image = cv2.imread(input_file)
        input_bg_color = input_image[0, 0, :]
        # Calculate the absolute difference between the background color and the image
        abs_diff = cv2.absdiff(input_image, input_bg_color)
        gray_diff = cv2.cvtColor(abs_diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_diff, 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            x1, y1, x2, y2 = x, y, x + w, y + h
            bboxes.append([x1, y1, x2, y2])
        else:
            bboxes.append(None)
    # return an empty list if all elements are None
    if all([bbox is None for bbox in bboxes]):
        return []
    return bboxes


def scale_image(image_path, size_threshold = 0.7):
    head_image = cv2.imread(image_path)
    w, h = head_image.shape[:2]
    head_box = get_bbox_from_contours([image_path])
    logger.info("head_box: %s", head_box)
    logger.info("w: %s, h: %s", w, h)
    if not head_box:
        return None
    x1, y1, x2, y2 = head_box[0]
    head_width = x2 - x1
    head_height = y2 - y1
    logger.info("head_width: %s, head_height: %s", head_width, head_height)

    if head_height > size_threshold * h or head_width > size_threshold * w:
        logger.info("Scaling the image....")
        scale_factor = size_threshold * w / head_width if head_width > head_height else size_threshold * h / head_height
        logging.info("scale_factor: %s", scale_factor)
        new_head_image = cv2.resize(
            head_image, (int(w * scale_factor), int(h * scale_factor))
        )
        logger.info("new_head_image.shape: %s", new_head_image.shape)
        # pad the image to 512x512
        background_color = head_image[0, 0, :]
        # generate a new image with background color and using w and h as images's width and height
        new_image = np.zeros((w, h, 3), np.uint8)
        new_image[:, :] = background_color
        # put the image at the center
        x_offset = (512 - new_head_image.shape[1]) // 2
        y_offset = (512 - new_head_image.shape[0]) // 2
        new_image[y_offset : y_offset + new_head_image.shape[0], x_offset : x_offset + new_head_image.shape[1]] = new_head_image
        # rewrite the image
        new_path = image_path.replace(".png", "_scaled.png")
        logger.info("saving the scaled image to %s", new_path)
        cv2.imwrite(new_path, new_image)
        return new_path
    return None


def scale_image_and_landmarks(image_path, landmark_path):
    # check if the landmarks are too close to the boundary
    landmarks = read_openpose(landmark_path)[0]
    min_y = min([landmark[1] for landmark in landmarks])
    max_y = max([landmark[1] for landmark in landmarks])
    gap_top = min_y
    gap_bottom = 1 - max_y
    if gap_bottom < 0.1 and gap_top > 0.3:
        # move all the landmarks up and redraw the image
        # TODO: NEED to redraw the image, not just put points to image
        landmarks = np.array(
            [(landmark[0], landmark[1] - 0.2, 1.0) for landmark in landmarks]
        )
        # save the landmarks
        write_openpose(landmark=landmarks, bbox=None, json_path=landmark_path)
        # draw images
        new_img = np.zeros((512, 512, 3), np.uint8)
        for landmark in landmarks:
            x, y, _ = landmark
            x = int(x * 512)
            y = int(y * 512)
            new_img = cv2.circle(new_img, (x, y), 1, (255, 255, 255), -1)
        cv2.imwrite(image_path.as_posix(), new_img)


def generate_images_from_one_control_input(
    control_net_img_path: Path,
    port=7860,
    num_prompts=10,
    batch_size=2,
    steps=30,
    bald_rate=0.9,
    generate_landmarks=False,
    control_net_model=None,
    preprocessor=None,
):
    ref_name = control_net_img_path.stem
    logging.info(f"input image: {control_net_img_path}, ref_name: {ref_name}")
    output_dir = Path("./avatar_generation/outputs/") / f"{ref_name}"
    # logger.info("Generating output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    # generate control json
    logger.info("Generating initial landmark json.")
    control_json_path = output_dir / "pose.json"
    if generate_landmarks:
        print("generate_landmarks")
        get_pose_estimation(
            control_net_img_path.as_posix(), control_json_path.as_posix()
        )
    else:
        print("do not generate json using stable diffusion's api")
        # find the json file and copy, json file is at the same folder of input image but with name _landmark.json
        # instead of _controlInput.png
        if "_controlInput" in control_net_img_path.stem:
            src_control_net_json_path = (
                control_net_img_path.parent
                / f'{control_net_img_path.stem.replace("_controlInput","_landmark")}.json'
            )
        else:
            src_control_net_json_path = (
                control_net_img_path.parent / f"{control_net_img_path.stem}_landmark.json"
            )
        logging.info(f"src_control_net_json_path: {src_control_net_json_path}, control_json_path: {control_json_path}")
        # shutil.copy(src_control_net_json_path, control_json_path)

    # scale the image if it is too close to the boundary
    scaled_path = scale_image(control_net_img_path.as_posix())
    if scaled_path:
        control_net_img_path = Path(scaled_path)

    # scale_image_and_landmarks(control_net_img_path, control_json_path)

    # generate prompt
    generated_prompts = set()
    # logger.info("Generating prompts.")
    prompt_list = []
    image_list = []
    count = 0
    print(f"copying control_net_img_path {control_net_img_path} to output_dir")
    shutil.copy(control_net_img_path, output_dir / "input_ori.png")
    for _ in range(num_prompts):
        prompt = generate_unique_prompt(generated_prompts, bald_rate)
        if "_40" or "_320" in control_net_img_path.stem:
            prompt += " Generate side view head of the character."
        logger.info("Generated prompt: %s", prompt)

        control_net = ControlnetRequest(
            prompt=prompt,
            path=control_net_img_path.as_posix(),
            batch_size=batch_size,
            steps=steps,
            control_net_checkpoint=control_net_model,
            preprocess_module=preprocessor,
        )
        control_net.build_body()
        output = control_net.send_request()
        result = output["images"]
        # logger.info("Received %d images.", len(result) - 1)
        for idx, item in enumerate(result):
            res_img = Image.open(io.BytesIO(base64.b64decode(item.split(",", 1)[0])))
            if idx == len(result) - 1:
                save_path = (output_dir / "input_control.png").as_posix()
            else:
                save_path = (output_dir / f"avatar_{count}.png").as_posix()
                prompt_list.append(prompt)
                image_list.append(save_path)
                count += 1
            res_img.save(save_path)

    with open(output_dir / "prompt_image.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(zip(prompt_list, image_list))


def stop_port(port):
    url = f"http://localhost:{port}/sdapi/v1/server-kill"
    try:
        response = requests.post(url)
        logger.info(
            "Stopping service on port %s. status %s, response %s",
            port,
            response.status_code,
            response.json(),
        )
    except requests.exceptions.ConnectionError:
        logger.error("Failed to stop service on port %s.", port)


def stop_all_services(ports):
    for port in ports:
        stop_port(port)


def generate_image_wrapper(**kwargs):
    input_list = kwargs.get("input_list")
    port = kwargs.get("port")
    num_prompts = kwargs.get("num_prompts")
    batch_size = kwargs.get("batch_size")
    steps = kwargs.get("steps")
    bald_rate = kwargs.get("bald_rate")
    control_net_model = kwargs.get("control_net_model")
    preprocessor = kwargs.get("preprocessor")
    if control_net_model is None or "openpose" in control_net_model:
        logging.info("control_net_model is None or openpose")
        for input_path in input_list:
            logging.info(f"input_path: {input_path}")
            generate_images_from_one_control_input(
                input_path, port, num_prompts, batch_size, steps, bald_rate
            )
    else:
        logging.info("control_net_model is not None and not openpose")
        for input_path in input_list:
            logging.info(f"input_path: {input_path}")
            generate_images_from_one_control_input(
                control_net_img_path=input_path,
                port=7860,
                num_prompts=num_prompts,
                batch_size=batch_size,
                steps=steps,
                bald_rate=bald_rate,
                generate_landmarks=False,
                control_net_model=control_net_model,
                preprocessor=preprocessor,
            )


def run_batch_file(port):
    cmd = [
        "cmd.exe",
        "/c",
        f"start /min webui.bat --api --port {port} --skip-install --api-server-stop",
    ]
    # command = ['cmd.exe', '/c', 'start', 'cmd.exe', '/  k', 'path_to_your_batch_file.bat']
    subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    num_port = 1
    ports = [7860 + i for i in range(num_port)]
    number_prompts = 20
    batch_size = 4
    steps = 60

    input_dir = Path("./avatar_generation/inputs")
    # input_list = list(input_dir.glob("*_controlInput.png"))

    # bald_rate = 0.9
    # # TODO: check if the server is ready if not wait until it is ready
    # generate_image_wrapper((input_list, ports[0], number_prompts, batch_size, steps, bald_rate))

    input_list = list(input_dir.glob("*.png"))
    depth_control_avatars_keywords = [
        "Nendoroid",
        "Chibi",
        "mermaid",
        "elongatedGirl",
        "bionicGirl",
        "animeBasemesh",
        "anya",
    ]
    input_list = [
        filename
        for filename in input_list
        if ("_controlInput" not in filename.stem) and ("_scaled" not in filename.stem)
        and (
            any(keyword in filename.stem for keyword in depth_control_avatars_keywords)
        )
    ]
    print("input_list", input_list)
    bald_rate = 0.9
    control_net_model = "lllyasvielsd-controlnet-depth"
    preprocess = "depth-midas"
    generate_image_wrapper(
        input_list=input_list,
        num_prompts=number_prompts,
        batch_size=batch_size,
        steps=steps,
        bald_rate=bald_rate,
        control_net_model=control_net_model,
        preprocess="depth_midas",
    )
