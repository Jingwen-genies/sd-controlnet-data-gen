import base64
import json
from typing import List, Tuple, Union

import boto3
import cv2
import numpy as np
from PIL import Image


def np_array_to_base64(img: np.ndarray) -> str:
    """
    Convert a numpy.ndarray image to a base64-encoded string.
    """
    _, buffer = cv2.imencode(".jpg", img)
    encoded_image = base64.b64encode(buffer).decode("utf-8")
    return encoded_image


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The base64-encoded image as a string.
    """
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_image


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """
    Get the dimensions (width and height) of an image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        Tuple[int, int]: A tuple containing the width and height of the image.
    """

    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def create_json_request(
    paths: Union[str, List[str]] = [],
    radius: int = 3,
    show_kpt_idx: bool = False,
) -> str:
    """
    Create a JSON structure containing base64 encoded data and sizes of images.

    Args:
        paths (Union[str, List[str]]): Either a single path string or a list of paths.
        raduis: The radius of the keypoint dot in visualization
        show_kpt_idx: switch on whether to show the labels for each keypoint on visualization
        font_size: font size of the keypoint labels in the visualization images

    Returns:
        Dict[str, Dict]: Dictionary containing base64 encoded data and sizes of the images.
    """
    result = {
        "visualization_options": {
            "radius": radius,
            "show_kpt_idx": show_kpt_idx,
        },
        "input_images": {},
    }
    # Ensure paths is a list even if a single string is provided
    if isinstance(paths, str):
        paths = [paths]
    if len(paths) != 0:
        for path in paths:
            encoded = encode_image_to_base64(path)
            width, height = get_image_dimensions(path)
            result["input_images"][path] = {
                "image": encoded,
                "size": {"width": width, "height": height},
            }

    # Dump data to the specified JSON file
    return json.dumps(result, indent=4)


def get_landmarks_from_response(response: dict) -> List[np.ndarray]:
    """
    Get the landmarks from the response of the SageMaker endpoint.

    Args:
        response (dict): The response from the SageMaker endpoint.

    Returns:
        List[np.ndarray]: A list of landmarks.
    """
    data = json.loads(response["Body"].read().decode("utf-8"))
    data = json.loads(data["result"])
    r = data["data"]["result"]
    k = next(iter(r))
    landmarks = json.loads(r[k])["predictions"]
    return [np.array(landmark) for landmark in landmarks]


if __name__ == "__main__":
    # # SageMaker
    runtime_sm_client = boto3.client(service_name="sagemaker-runtime")
    # image_paths = ["test_data/girl.png", "test_data/girl2.png"]
    image_paths = ["test_data/animeBasemesh_head_view_0.png"]
    payload = create_json_request(paths=image_paths, radius=3, show_kpt_idx=True)

    # endpoint_name = "facial-landmark-app-v1"
    endpoint_name = "facial-landmark-app-v2"
    results = runtime_sm_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload,
    )
    content = json.loads(results["Body"].read().decode("utf-8"))
    content = json.loads(content["result"])
    status = content["status"]
    message = content["message"]
    print(message)
    r = content["data"]["result"]
    k = next(iter(r))
    print(json.loads(r[k])["predictions"][0][0])