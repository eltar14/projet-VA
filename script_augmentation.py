import cv2
import os
import numpy as np
import albumentations as A

IMAGES_SRC_PATH = "test/images"
LABELS_SRC_PATH = "test/labels"
IMAGES_DEST_PATH = "test/train_augmented/images"
LABELS_DEST_PATH = "test/train_augmented/labels"

# IMAGES_SRC_PATH = "YOLO_TRAIN/datasets/train/images"
# LABELS_SRC_PATH = "YOLO_TRAIN/datasets/train/labels"
# IMAGES_DEST_PATH = "YOLO_TRAIN/datasets/train_augmented/images"
# LABELS_DEST_PATH = "YOLO_TRAIN/datasets/trainç_augmented/labels"

# Define the augmentation pipeline
def augment_image(image:np.ndarray, bbox:list[float], image_size:int=416):
    """
    Augment an image and its bounding box.

    Args:
        image (np.ndarray): Image to augment.
        bbox (list[float]): Bounding box to augment.
        image_size (int, optional): Image size to resize. Defaults to 416.
    """
    
    horizontal_flip = True
    augmentation_pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.1),
        A.CLAHE(clip_limit=4.0, p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, p=0.5),
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.RandomCrop(p=0.5, height=image_size, width=image_size),
    ], bbox_params=A.BboxParams(format='yolo'))
    
    transformed_tuple = augmentation_pipeline(image=image, bboxes=[bbox])
    transformed_image, transformed_bboxes = transformed_tuple['image'], transformed_tuple['bboxes']
    return transformed_image, transformed_bboxes


def load_images_and_labels():
    # Get the current working directory
    current_working_dir = os.getcwd()
    
    # Concatenate the current working directory with the source paths
    images_source_path = os.path.join(current_working_dir, IMAGES_SRC_PATH)
    labels_source_path = os.path.join(current_working_dir, LABELS_SRC_PATH)
    
    # If one the source paths is not a directory, raise an error
    if not os.path.isdir(images_source_path) or not os.path.isdir(labels_source_path):
        raise ValueError("Source paths must be directories")

    # Get the directory listing of the source paths
    image_files = os.listdir(images_source_path)
    label_files = os.listdir(labels_source_path)
    
    # Iterate over all images and labels in their own directories, loading them one by one to not run out of memory
    for image_path, label_path in zip(image_files, label_files):
        image = cv2.imread(os.path.join(images_source_path,image_path))
        label = np.loadtxt(os.path.join(labels_source_path,label_path))
        
        # Transform the array to a list
        label = label.tolist()

        # Remove the first column of the label, as it is the class index
        class_element = label[0]
        label = label[1:]
        label.append(class_element)

        transformed_image, transformed_bbox = augment_image(image=image, bbox=label)
        
        # Extract the file name of both paths
        image_name = os.path.basename(image_path)
        label_name = os.path.basename(label_path)
        
        # Create the destination paths
        image_path = os.path.join(IMAGES_DEST_PATH, image_name)
        label_path = os.path.join(LABELS_DEST_PATH, label_name)
        
        # Save the transformed image and label
        cv2.imwrite(image_path, transformed_image)
        np.savetxt(label_path, transformed_bbox)
        
load_images_and_labels()