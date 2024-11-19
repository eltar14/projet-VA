import os
import shutil
from tqdm import tqdm

BASE_PATH = "C:\\Users\\lebou\\OneDrive - yncréa\\000_UQAC\\Cours Automne\\Vision artificielle\\datasets\\ASL_Dataset"

OUTPUT_PATH = "C:\\Users\\lebou\\OneDrive - yncréa\\000_UQAC\\Cours Automne\\Vision artificielle\\datasets\\ASL_Dataset_YOLO_3"

os.makedirs(OUTPUT_PATH, exist_ok=True)

all_classes = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "Nothing", "O", "P", "Q", "R", "S",
    "Space", "T", "U", "V", "W", "X", "Y", "Z"
]

excluded_classes = ["Space", "Nothing"]

classes = [cls for cls in all_classes if cls not in excluded_classes]  # classes retenues
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}  # indexes


# names: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# print(class_to_idx)
print(f"names: ['{'\', \''.join(classes)}']") # les classes pour le fichier yaml


def process_dataset(split):
    split_path = os.path.join(BASE_PATH, split)

    split_output_path = os.path.join(OUTPUT_PATH, split.lower()) # creer les dossier train test ...
    images_path = os.path.join(split_output_path, "images")
    labels_path = os.path.join(split_output_path, "labels")
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    for class_name in tqdm(os.listdir(split_path), desc=f"Processing {split}", unit="class"): # pour chaque dossier

        if class_name not in classes: # si le nom du dossier est PAS dans les classes selectionnées on ignore
            continue

        class_path = os.path.join(split_path, class_name)
        if not os.path.isdir(class_path):
            continue

        class_idx = class_to_idx[class_name]
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # vérif format
                new_img_name = f"{split}_{class_name}_{img_name}"  # noms uniques
                new_img_name = new_img_name.replace(" ", "_")  # remplace les espaces pour éviter les erreurs
                new_img_path = os.path.join(images_path, new_img_name)

                shutil.copy(img_path, new_img_path)

                label_path = os.path.join(labels_path, f"{os.path.splitext(new_img_name)[0]}.txt") # label correspondant
                with open(label_path, 'w') as label_file:
                    label_file.write(f"{class_idx} 0.5 0.5 1 1\n") # en dur parce que flemme


if __name__ == "__main__":
    for split in ["Train", "Test"]:  # titres des dossiers à traiter
        process_dataset(split)

    print("FINI !.")
