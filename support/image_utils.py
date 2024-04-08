"""
Utility functions that deal with bounding box of the facial landmark detection
"""
import cv2
import numpy as np


def valid_bbox(bbox, image_width, image_height):
    """
    check if the bounding box provided is a valid bbox, basically call a hypositesis test check if this bbox can be
    drawned from the bbox distribution with high probability
    Args:
        x1: left top corner x
        y1: left top corner y
        x2: right bottom corner x
        y2: right bottom corner y
        w: width of the image
        h: height of the image
    Returns: boolean
    """
    x1, y1, x2, y2 = bbox
    bbox_center_height = (y1 + y2) // 2
    bbox_center_width = (x1 + x2) // 2
    bbox_area = (x2 - x1) * (y2 - y1)
    image_area = image_width * image_height
    bbox_height = y2 - y1
    bbox_width = x2 - x1
    # TODO: change to use the statistics of the testing data in the future
    too_small = (
        bbox_area < 0.25 * image_area
        or bbox_width < 0.5 * image_width
        or bbox_height < 0.5 * image_height
    )
    too_large = (
        bbox_area > 0.75 * image_area
        or bbox_width > 0.8 * image_width
        or bbox_height > 0.8 * image_height
    )
    too_high = bbox_center_height < image_height / 2.5
    too_left = bbox_center_width < image_width / 5
    too_right = bbox_center_width > image_width / 5 * 4

    if too_small or too_large or too_high or too_left or too_right:
        return False
    return True


def plot_bbox(image, bboxes, save_path):
    """
    Plot the bbox on image.
    """
    for bbox in bboxes:
        x1, y1, x2, y2, score = bbox
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        left_top = np.array([int(x1), int(y1)])
        right_bottom = np.array([int(x2), int(y2)])
        image = np.array(image)
        image = cv2.rectangle(
            img=image, pt1=left_top, pt2=right_bottom, color=(0, 255, 0), thickness=2
        )

        def clip(value, min_value, max_value):
            return max(min_value, min(max_value, value))

        text_x = clip(int(x1), 0, 511)
        text_y = clip(
            int(y1) + 50, 0, 511
        )  # Adjust this value to control the vertical position

        text = f"Prob: {score:.2f}, Area: {area}"
        cv2.putText(
            image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
        )
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path, image_rgb)


def find_bounding_box_from_contours(input_image):
    input_bg_color = input_image[0, 0, :]
    # Calculate the absolute difference between the background color and the image
    abs_diff = cv2.absdiff(input_image, input_bg_color)
    gray_diff = cv2.cvtColor(abs_diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_diff, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        # cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow('Avatar Bounding Box', input_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        x1, y1, x2, y2 = x, y, x + w, y + h
        y1 += 125
        return [[x1, y1, x2, y2]]
    else:
        return None


def calculate_head_to_image_ratio(bounding_box, image_size):
    """
    Calculate the ratio of the head to the image
    """
    bbox_width = bounding_box[2] - bounding_box[0]
    bbox_height = bounding_box[3] - bounding_box[1]
    width_ratio = bbox_width / image_size[0]
    height_ratio = bbox_height / image_size[1]
    return width_ratio, height_ratio


def get_adjusted_image_size(bounding_box, ref_width_ratio, ref_height_ratio):
    """
    Get the adjusted image size based on the reference ratio
    """
    bbox_width = bounding_box[2] - bounding_box[0]
    bbox_height = bounding_box[3] - bounding_box[1]
    adjusted_width = int(bbox_width / ref_width_ratio)
    adjusted_height = int(bbox_height / ref_height_ratio)
    return adjusted_width, adjusted_height


def scale_image(input_image, ref_image_path):
    """
    Scale the input image so that the ratio of the head to the image is the same as the reference image
    """
    bg_color = input_image[0, 0, :]
    reference_image = cv2.imread(ref_image_path)
    input_bbox = find_bounding_box_from_contours(input_image)[0]
    ref_bbox = find_bounding_box_from_contours(reference_image)[0]

    ref_width_ratio, ref_height_ratio = calculate_head_to_image_ratio(ref_bbox, reference_image.shape)

    adjusted_width, adjusted_height = get_adjusted_image_size(input_bbox, ref_width_ratio, ref_height_ratio)

    # Decide whether to crop or pad based on the comparison of adjusted and current dimensions
    if adjusted_width > input_image.shape[1] or adjusted_height > input_image.shape[0]:
        # Pad the image because the adjusted dimensions are larger than the current image size
        result_image = pad_image_to_size(input_image, adjusted_width, adjusted_height, bg_color.tolist())
    else:
        # Crop the image because the adjusted dimensions are smaller than the current image size
        result_image = crop_image_to_size(input_image, adjusted_width, adjusted_height)
    # pad the image to a square one
    if result_image.shape[0] != result_image.shape[1]:
        result_image = pad_image_to_size(result_image, max(result_image.shape), max(result_image.shape), bg_color.tolist())
    # resize
    result_image = cv2.resize(result_image, (512, 512))
    return result_image


def pad_image_to_size(image, target_width, target_height, bg_color=(92, 92, 92)):
    # This function will add padding to the image to reach the target size
    height, width = image.shape[:2]
    padding_top = max((target_height - height), 0)
    padding_left = max((target_width - width) // 2, 0)
    padding_right = max(target_width - width - padding_left, 0)
    padded_image = cv2.copyMakeBorder(image, padding_top, 0, padding_left, padding_right,
                                      cv2.BORDER_CONSTANT, value=bg_color)
    return padded_image


def crop_image_to_size(image, target_width, target_height):
    # This function will crop the image to the target size, focusing on the center
    height, width = image.shape[:2]
    start_x = (width - target_width) // 2
    start_y = (height - target_height) // 2
    cropped_image = image[start_y:start_y + target_height, start_x:start_x + target_width]
    return cropped_image


