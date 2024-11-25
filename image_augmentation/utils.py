import albumentations as A
import cv2
import os
import yaml
import numpy as np
#import pybboxes as pbx

# Import the constants file
with open("constants.yaml", 'r') as stream:
    CONSTANTS = yaml.safe_load(stream)


def is_image_by_extension(file_name:str)->bool:
    """
    Check if the given file has a recognized image extension.

    Args:
        file_name (str): Name of the file.

    Returns:
        bool: True if the file has a recognized image extension, False otherwise.

    """
    # List of common image extensions
    image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']
    # Get the file extension
    file_extension = file_name.lower().split('.')[-1]
    # Check if the file has a recognized image extension
    return file_extension in image_extensions


def get_input_data(img_file:str):
    """
    Get input data for image processing.

    Args:
        img_file (str): Name of the input image file.

    Returns:
        tuple: A tuple containing the image, ground truth bounding boxes, and augmented file name.

    """
    # Read the input image
    file_name = os.path.splitext(img_file)[0]
    aug_file_name = f"{file_name}_{CONSTANTS["TRANSFORMED_FILE_NAME"]}"
    image = cv2.imread(os.path.join(CONSTANTS["INPUT_IMAGE_PATH"], img_file))
    
    # Read the bounding boxes labels
    label_path = os.path.join(CONSTANTS["INPUT_LABEL_PATH"], f"{file_name}.txt")
    gt_bboxes = get_bboxes_list(label_path, CONSTANTS['CLASSES'])
    return image, gt_bboxes, aug_file_name


def single_extract_bbox_informations(yolo_bbox:str, class_names:list)->list:
    """
    Extracts bounding box information for a single object from YOLO format.

    Args:
        yolo_bbox (str): YOLO format string representing bounding box information.
        class_names (list): List of class names corresponding to class numbers.

    Returns:
        list: A list containing [x_center, y_center, width, height, class_name].
    """
    # Split elements of the label
    str_bbox_list = yolo_bbox.split()
    
    # Extract class number, class name, and bounding box values
    class_number = int(str_bbox_list[0])
    class_name = class_names[class_number]
    bbox_values = list(map(float, str_bbox_list[1:]))
    
    # Convert bounding box values to x_center, y_center, width, height, class_name for albumentions
    albumentions_bb = bbox_values + [class_name]
    return albumentions_bb


def multiple_extract_bbox_informations(yolo_str_labels:str, classes:list)->list:
    """
    Extracts bounding box information for multiple objects from YOLO format.

    Args:
        yolo_str_labels (str): YOLO format string containing bounding box information for multiple objects.
        classes (list): List of class names corresponding to class numbers.

    Returns:
        list: A list of lists, each containing [x_center, y_center, width, height, class_name].
    """
    albumentions_bb_lists = []
    yolo_list_labels = yolo_str_labels.split('\n')
    for yolo_str_label in yolo_list_labels:
        if yolo_str_label:
            albumentions_bb_list = single_extract_bbox_informations(yolo_str_label, classes)
            albumentions_bb_lists.append(albumentions_bb_list)
    return albumentions_bb_lists


def get_bboxes_list(input_label_path:str, classes:list)->list:
    """
    Reads YOLO format labels from a file and returns bounding box information.

    Args:
        input_label_path (str): Path to the YOLO format labels file.
        classes (list): List of class names corresponding to class numbers.

    Returns:
        list: A list of lists, each containing [x_center, y_center, width, height, class_name].
    """
    # Read the YOLO format labels
    yolo_str_labels = open(input_label_path, "r").read()

    if not yolo_str_labels:
        print("No object")
        return []
    
    # If there are multiple bboxes, extract information for each object, otherwise extract information for a single object
    lines = [line.strip() for line in yolo_str_labels.split("\n") if line.strip()]
    albumentions_bb_lists = multiple_extract_bbox_informations("\n".join(lines), classes) if len(lines) > 1 else [single_extract_bbox_informations("\n".join(lines), classes)]

    return albumentions_bb_lists


def single_convert_bbox_yolo(transformed_bboxes:list, class_names:list)->list:
    """
    Convert bounding boxes for a single object to YOLO format.

    Parameters:
    - transformed_bboxes (list): Bounding box coordinates and class name.
    - class_names (list): List of class names.

    Returns:
    - list: Bounding box coordinates in YOLO format.
    """
    if transformed_bboxes:
        class_number = class_names.index(transformed_bboxes[-1])
        bboxes = list(transformed_bboxes)[:-1]
        bboxes.insert(0, class_number)
    else:
        bboxes = []
    return bboxes


def multi_obj_bb_yolo_conversion(augmented_labels:list, class_names:list)->list:
    """
    Convert bounding boxes for multiple objects to YOLO format.

    Parameters:
    - aug_labels (list): List of bounding box coordinates and class names.
    - class_names (list): List of class names.

    Returns:
    - list: List of bounding box coordinates in YOLO format for each object.
    """
    yolo_labels = [single_convert_bbox_yolo(augmented_label, class_names) for augmented_label in augmented_labels]
    return yolo_labels


def save_augmented_labels(transformed_bboxes:list, label_path:str, label_name:str):
    """
    Save augmented bounding boxes to a label file.

    Args:
        transformed_bboxes (list): List of augmented bounding boxes.
        label_path (str): Path to the output label directory.
        label_name (str): Name of the label file.

    """
    label_output_path = os.path.join(label_path, label_name)
    with open(label_output_path, 'w') as output:
        for bbox in transformed_bboxes:
            updated_bbox = str(bbox).replace(',', ' ').replace('[', '').replace(']', '')
            output.write(updated_bbox + '\n')


def save_augmented_image(transformed_image:np.ndarray, output_image_path:str, img_name:str):
    """
    Save augmented image to an output directory.

    Args:
        transformed_image (numpy.ndarray): Augmented image.
        OUTPUT_IMAGE_PATH (str): Path to the output image directory.
        img_name (str): Name of the image file.

    """
    output_img_path = os.path.join(output_image_path, img_name)
    cv2.imwrite(output_img_path, transformed_image)


# def draw_yolo(image, labels, file_name):
#     """
#     Draw bounding boxes on an image based on YOLO format.

#     Args:
#         image (numpy.ndarray): Input image.
#         labels (list): List of bounding boxes in YOLO format.

#     """
#     H, W = image.shape[:2]
#     for label in labels:
#         yolo_normalized = label[1:]
#         box_voc = pbx.convert_bbox(tuple(yolo_normalized), from_type="yolo", to_type="voc", image_size=(W, H))
#         cv2.rectangle(image, (box_voc[0], box_voc[1]),
#                       (box_voc[2], box_voc[3]), (0, 0, 255), 1)
#     cv2.imwrite(f"bb_image/{file_name}.png", image)
#     # cv2.imshow(f"{file_name}.png", image)
#     # cv2.waitKey(0)

def has_negative_element(lst:list)->bool:
    """
    Check if the given list contains any negative element.

    Args:
        lst (list): List of elements.

    Returns:
        bool: True if there is any negative element, False otherwise.
    """
    return any(n < 0 for n in lst)


def get_augmented_results(image:np.ndarray, bboxes:list)->tuple:
    """
    Apply data augmentation to an input image and bounding boxes.

    Parameters:
    - image (numpy.ndarray): Input image.
    - bboxes (list): List of bounding boxes in YOLO format [x_center, y_center, width, height, class_name].

    Returns:
    - tuple: A tuple containing the augmented image and the transformed bounding boxes.
    """

    # Define the augmentations
    transform = A.Compose([
        A.RandomCrop(width=300, height=300, p=CONSTANTS["RANDOM_CROP_PROB"]),
        A.HorizontalFlip(p=CONSTANTS["HORIZONTAL_FLIP_PROB"]),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=CONSTANTS["RANDOM_BRIGHTNESS_CONTRAST_PROB"]),
        A.CLAHE(clip_limit=1, tile_grid_size=(8, 8), always_apply=True, p=CONSTANTS["CLAHE_PROB"]),
        A.Resize(300, 300, p=CONSTANTS["RESIZE_PROB"]),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=CONSTANTS["HUE_SATURATION_VALUE_PROB"]),
        A.GaussianBlur(blur_limit=(3,7), p=CONSTANTS["GAUSSIAN_BLUR_PROB"]),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=CONSTANTS["SHIFT_SCALE_ROTATE_PROB"]),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, p=CONSTANTS["COARSE_DROPOUT_PROB"]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=CONSTANTS["NORMALIZE_PROB"])
        ], bbox_params=A.BboxParams(format='yolo'))

    # Apply the augmentations
    transformed = transform(image=image, bboxes=bboxes)
    transformed_image, transformed_bboxes = transformed['image'], transformed['bboxes']
    
    return transformed_image, transformed_bboxes


def has_negative_element(matrix:list[list])->bool:
    """
    Check if there is a negative element in the 2D list of augmented bounding boxes.

    Args:
        matrix (list[list]): The 2D list.

    Returns:
        bool: True if a negative element is found, False otherwise.

    """
    return any(element < 0 for row in matrix for element in row)


def save_augmentation(transformed_image:np.ndarray, transform_bboxes:list, transformed_file_name:str)->None:
    """
    Saves the augmented label and image if no negative elements are found in the transformed bounding boxes.

    Parameters:
        transformed_image (numpy.ndarray): The augmented image.
        transform_bboxes (list): The transformed bounding boxes.
        trans_file_name (str): The name for the augmented output.

    Returns:
        None
    """
    tot_objs = len(transform_bboxes)
    if tot_objs:
        # Convert bounding boxes to YOLO format
        transform_bboxes = multi_obj_bb_yolo_conversion(transform_bboxes, CONSTANTS['CLASSES']) if tot_objs > 1 else [single_convert_bbox_yolo(transform_bboxes[0], CONSTANTS['CLASSES'])]
        if not has_negative_element(transform_bboxes):
            # Save augmented label and image
            save_augmented_labels(transform_bboxes, CONSTANTS["OUTPUT_LABEL_PATH"], transformed_file_name + ".txt")
            save_augmented_image(transformed_image, CONSTANTS["OUTPUT_IMAGE_PATH"], transformed_file_name + ".png")
            # Draw bounding boxes on the augmented image
            #draw_yolo(trans_image, trans_bboxes, trans_file_name)
        else:
            print("Found Negative element in Transformed Bounding Box...")
    else:
        print("Label file is empty")

def create_folder(folder_paths:list[str]):
    """
    Create folders if they do not exist

    Args:
        folder_paths (list[str]): List of folder paths.
    """
    
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder created: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")