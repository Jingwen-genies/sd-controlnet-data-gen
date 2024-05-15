import base64
import io
import os

import logging

logger = logging.getLogger(__name__)
import cv2
import requests
import rootutils
from PIL import Image

project_root = rootutils.setup_root(
    search_from=__file__,
    indicator=".project-root",
    dotenv=True,
    pythonpath=True,
    cwd=False,
)
os.chdir(project_root)
from avatar_generation.image_generation.prompt import negative_prompt, positive_prompt

"""
    To use this example make sure you've done the following steps before executing:
    1. Ensure automatic1111 is running in api mode with the controlnet extension. 
       Use the following command in your terminal to activate:
            ./webui.sh --no-half --api
    2. Validate python environment meet package dependencies.
       If running in a local repo you'll likely need to pip install cv2, requests and PIL 
"""


class ControlnetRequest:
    def __init__(
        self,
        prompt,
        path,
        batch_size=1,
        steps=20,
        sd_model_checkopint="v1-5-pruned-emaonly.safetensors",
        control_net_checkpoint="control_v11p_sd15_openpose",
        preprocess_module=None,
    ):
        self.url = f"http://stablediffusion.dev.genies.com:7860/sdapi/v1/txt2img"
        self.prompt = prompt
        self.img_path = path
        self.body = None
        self.batch_size = batch_size
        self.steps = steps

        self.sd_model_checkpoint = sd_model_checkopint
        self.control_net_checkpoint = control_net_checkpoint
        self.preprocess_module = preprocess_module

    def build_body(self):
        self.body = {
            "prompt": self.prompt + " " + positive_prompt,
            "negative_prompt": negative_prompt,
            "batch_size": self.batch_size,
            "steps": self.steps,
            "cfg_scale": 7,
            "sampling_method": "Euler a",
            "override_settings": {
                "sd_model_checkpoint": self.sd_model_checkpoint,
            },
            # "alwayson_scripts": {
            #     "controlnet": {
            #         "args": [
            #             {
            #                 "enabled": True,
            #                 "module": self.preprocess_module,
            #                 "model": self.control_net_checkpoint,
            #                 "weight": 1.0,
            #                 "image": self.read_image(),
            #                 "resize_mode": 1,
            #                 "lowvram": False,
            #                 "processor_res": 512,
            #                 "threshold_a": 64,
            #                 "threshold_b": 64,
            #                 "guidance_start": 0.0,
            #                 "guidance_end": 1.0,
            #                 "control_mode": 0,
            #                 "pixel_perfect": False,
            #             },
            #             {
            #                 "enabled": True,
            #                 "module": "canny",
            #                 "model": "lllyasvielsd-controlnet-canny",
            #                 "weight": 1.0,
            #                 "image": self.read_image(),
            #                 "resize_mode": 1,
            #                 "lowvram": False,
            #                 "processor_res": 512,
            #                 "threshold_a": 64,
            #                 "threshold_b": 64,
            #                 "guidance_start": 0.0,
            #                 "guidance_end": 1.0,
            #                 "control_mode": 0,
            #                 "pixel_perfect": False,
                              "Control Weight": 0.6,
            #             }
            #
            #         ]
            #     }
            # },
        }

    def send_request(self):
        logger.debug("Sending request to controlnet...")
        response = requests.post(url=self.url, json=self.body)
        logger.info("Response status code: %s", response.status_code)
        return response.json()

    def read_image(self):
        img = cv2.imread(self.img_path)
        retval, bytes = cv2.imencode(".png", img)
        encoded_image = base64.b64encode(bytes).decode("utf-8")
        return encoded_image


if __name__ == "__main__":
    path = (
        "./avatar_generation/inputs/avatar-0001-animeBoy_head_view_0_controlInput.png"
    )
    prompt = "a girl with blond hair and blue eyes"

    control_net = ControlnetRequest(prompt, path)
    control_net.build_body()
    output = control_net.send_request()

    result = output["images"][0]

    image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
    image.show()
