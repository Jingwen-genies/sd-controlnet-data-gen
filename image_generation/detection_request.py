import io
import cv2
import base64
import requests
from PIL import Image
import rootutils
import os


project_root = rootutils.setup_root(search_from=__file__, indicator=".project-root", dotenv=True, pythonpath=True, cwd=False)
os.chdir(project_root)
from avatar_generation.image_generation.prompt import positive_prompt, negative_prompt


class OpenPoseDetectionRequest:
    def __init__(self, path,  port=7860):
        self.url = f"http://localhost:{port}/controlnet/detect"
        self.img_path = path
        self.body = None

    def build_body(self):
        self.body = {
        "controlnet_module": "openpose_faceonly",
        "controlnet_input_images": [self.read_image()],
        "controlnet_processor_res": 512,
        "controlnet_threshold_a": 64,
        "controlnet_threshold_b": 64,
        "low_vram": False
    }

    def send_request(self):
        response = requests.post(url=self.url, json=self.body)
        return response.json()

    def read_image(self):
        img = cv2.imread(self.img_path)
        retval, bytes = cv2.imencode('.png', img)
        encoded_image = base64.b64encode(bytes).decode('utf-8')
        return encoded_image


if __name__ == '__main__':
    path = './avatar_generation/inputs/blondWoman_head_view_0.png'

    detector = OpenPoseDetectionRequest(path)
    detector.build_body()
    output = detector.send_request()

    pose = output['poses'][0]
    print(pose)
    result = output['images'][0]

    image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
    image.show()
