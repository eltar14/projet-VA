from utils import *


def run_yolo_augmentor():
    """
    Run the YOLO augmentor on a set of images.

    This function processes each image in the input directory, applies augmentations,
    and saves the augmented images and labels to the output directories.

    """
    
    # create folders if they don't exist
    create_folder([CONSTANTS["OUTPUT_IMAGE_PATH"],CONSTANTS["OUTPUT_LABEL_PATH"]])
    
    images = [image for image in os.listdir(CONSTANTS["INPUT_IMAGE_PATH"]) if is_image_by_extension(image)]

    for image_number, image_file in enumerate(images):
        print(f"{image_number+1}-image is processing...\n")
        image, ground_truth_bboxes, augmented_file_name = get_input_data(image_file)
        augmented_image, augmented_label = get_augmented_results(image, ground_truth_bboxes)
        if len(augmented_image) and len(augmented_label):
            save_augmentation(augmented_image, augmented_label, augmented_file_name)

if __name__ == "__main__":
    run_yolo_augmentor()