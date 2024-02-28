import subprocess
import platform
import logging
import random
import os
from itertools import cycle
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
import traceback
import multiprocessing

warnings.filterwarnings("ignore", category=FutureWarning, module="pytorch_lightning.utilities.distributed", message=".*rank_zero_only has been deprecated.*")
project_root = rootutils.setup_root(search_from=__file__, indicator=".project-root", dotenv=True, pythonpath=True, cwd=False)
os.chdir(project_root)

from avatar_generation import (
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
    formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(handler)
    logger.addHandler(stream_handler)
    return logger


logger = setup_logging('./avatar_generation/logs.log')


def start_service(port=7860):
    # logger.info("Starting the WebUI service port %s.", port)
    # start the service based on OS
    operating_system = platform.system()
    if operating_system == "Windows":
        ext = ".bat"
        # logger.info("You are running Windows.")
    elif operating_system == "Linux":
        ext = ".sh"
        # logger.info("You are running Linux.")
    elif operating_system == "Darwin":
        ext = ".sh"
        # logger.info("You are running macOS.")
    else:
        logger.error(f"Unsupported operating system: {operating_system}")
        return None
    # logger.info(f"Starting the service using {ext} file.")
    try:
        cmd = [f"webui{ext}", "--api", "--port", f"{port}"]
        webui_process = subprocess.Popen(cmd, shell=True)
        return webui_process
    except OSError as e:
        logger.error(f"Failed to start the WebUI service: {e}")
        return None


import cv2
def read_image(image_path):
    img = cv2.imread(image_path)
    retval, bytes = cv2.imencode('.png', img)
    encoded_image = base64.b64encode(bytes).decode('utf-8')
    return encoded_image


def get_pose_estimation(input_image_path, save_path="./pose.json", port=7860):
    # logger.info("Generating controls from openpose.")
    logger.info("detecting pose from image %s on port %s", input_image_path, port)
    embedded_image = read_image(input_image_path)
    url = f"http://localhost:{port}/controlnet/detect"
    payload = {
        "controlnet_module": "openpose_faceonly",
        "controlnet_input_images": [embedded_image],
        "controlnet_processor_res": 512,
        "controlnet_threshold_a": 64,
        "controlnet_threshold_b": 64,
        "low_vram": False
    }
    response = requests.post(url, json=payload)
    output = response.json()
    pose = output['poses'][0]
    with open(save_path, "w") as f:
        json.dump(pose, f, indent=4)
    return response.json()


def generate_prompt():
    selected_object = random.choice(subject_list)
    selected_age = random.choice(age)
    selected_style = random.choice(style)
    selected_headwear = random.choice(headwear)
    selected_eye_color = random.choice(eye_color)
    selected_hair_color = random.choice(hair_color)
    selected_hair_style = random.choice(hair_style)
    selected_skin_color = random.choice(skin_color)
    selected_skin_texture = random.choice(skin_texture)
    selected_distinctive_marks = random.choice(distinguishing_marks)
    prompt = (f"A {selected_age} {selected_object} with {selected_style} style. The character should have "
              f"{selected_headwear} and {selected_eye_color} eyes. The hair should be {selected_hair_color} and "
              f"{selected_hair_style}. The skin should be {selected_skin_color} and {selected_skin_texture}. ")
    with_distinguishing_marks = random.random() > 0.7
    if with_distinguishing_marks:
        prompt += f"The character should have {selected_distinctive_marks}."
    return prompt


def generate_unique_prompt(history, max_attempts=100):
    for _ in range(max_attempts):
        prompt = generate_prompt()
        if prompt not in history:
            return prompt
    return None


def generate_images_from_one_control_input(control_net_img_path:Path, num_prompts=10, batch_size=2, steps=30, port=8000):
    # TODO: download json file from webui openpose editor?
    ref_name = control_net_img_path.stem
    output_dir = Path("./avatar_generation/outputs/") / f"{ref_name}_{port}"
    # logger.info("Generating output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    # generate control json
    # logger.info("Generating control json.")
    control_json_path = output_dir / "pose.json"
    get_pose_estimation(control_net_img_path.as_posix(), control_json_path.as_posix(), port=port)
    # generate prompt
    generated_prompts = set()
    # logger.info("Generating prompts.")
    prompt_list = []
    image_list = []
    count = 0
    shutil.copy(control_net_img_path, output_dir / "input_ori.png")
    for _ in range(num_prompts):
        prompt = generate_unique_prompt(generated_prompts)
        # logger.info("Generated prompt: %s", prompt)
        control_net = ControlnetRequest(
            prompt=prompt, path=control_net_img_path.as_posix(), batch_size=batch_size, steps=steps, port=port
        )
        control_net.build_body()
        output = control_net.send_request()
        result = output['images']
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


def is_service_ready(port):
    url = f"http://localhost:{port}/sdapi/v1/progress"
    try:
        response = requests.get(url)
        logger.info("Checking service status on port %s. status %s, response %s", port, response.status_code, response.json())
        if response.status_code == 200:
            progress_info = response.json()
            return progress_info["progress"] in [0, 1.0]
        return False
    except requests.exceptions.ConnectionError:
        return False


def get_next_port(ports, max_retries, retry_delay):
    for i in range(len(ports) + max_retries):
        port = random.choice(ports)
        if is_service_ready(port):
            return port
        time.sleep(retry_delay)
    return None


def start_services(ports):
    """
    Starts the webui service on the given ports
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


def stop_port(port):
    url = f"http://localhost:{port}/sdapi/v1/server-kill"
    try:
        response = requests.post(url)
        logger.info("Stopping service on port %s. status %s, response %s", port, response.status_code, response.json())
    except requests.exceptions.ConnectionError:
        logger.error("Failed to stop service on port %s.", port)


def stop_services(ports):
    for port in ports:
        stop_port(port)
    logger.info("Stopped webui services.")

#
# def generate_image_wrapper(args):
#     control_image_path, ports, num_prompts, batch_size, steps = args
#     try:
#         port = get_next_port(ports, max_retries=5, retry_delay=5)
#         generate_images_from_one_control_input(control_image_path, num_prompts, batch_size, steps, port)
#         logger.info("Successfully processed image %s. with port %s", control_image_path, port)
#         return control_image_path, True
#     except Exception as e:
#         logger.error("Error processing image %s: ",control_image_path)
#         logger.error(traceback.format_exc())
#         return control_image_path, False


def generate_image_wrapper(args):
    image_path_list, port, num_prompts, batch_size, steps = args
    for control_image_path in image_path_list:
        try:
            logger.info("Processing image %s.", control_image_path)
            generate_images_from_one_control_input(control_image_path, num_prompts, batch_size, steps, port)
            logger.info("Successfully processed image %s. with port %s", control_image_path, port)
            return control_image_path, True
        except Exception as e:
            logger.error("Error processing image %s: ",control_image_path)
            logger.error(traceback.format_exc())
            return control_image_path, False


def main():
    num_ports = 3
    num_prompts = 2
    batch_size = 1
    steps = 2
    ports = [7860 + i for i in range(num_ports)]
    logger.info("Starting services on ports %s.", ports)

    processes = start_services(ports)
    time.sleep(20)

    # load_balancer = LoadBalancer(ports, max_retries=5, retry_delay=5)

    input_dir = Path("./avatar_generation/inputs")
    image_list = list(input_dir.glob("*.png"))
    # split the image_list evenly into num_ports parts

    image_list_parts = [image_list[i::num_ports] for i in range(num_ports)]

    try:
        for i, image_list_part in enumerate(image_list_parts):
            port = ports[i]
            logger.info("Processing images on port %s.", port)
            processes = [multiprocessing.Process(target=generate_image_wrapper, args=((image_list_part, port, num_prompts, batch_size, steps),))]
            for process in processes:
                process.start()
    except Exception as e:
        logger.error("Error processing images: ")
        logger.error(traceback.format_exc())
    stop_services(ports)


if __name__ == "__main__":
    main()

