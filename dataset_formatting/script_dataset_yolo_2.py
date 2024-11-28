# =============================================================================
#   Fichier : script_dataset_yolo_2.py                                        #
#   Projet : projet-VA                                                        #
#   Auteur : Groupe Gabin Elise Margot Antoine                                #
#   Date : 21 novembre 2024                                                   #
#
#
# Pour mettre en forme le dataset suivant dans une arborescence spécifique et faciliter le merge de datasets
# Exclus les classes fn et sp, et vient remplacer les IDs de classe dans le fichier de label
# https://data.mendeley.com/datasets/xs6mvhx6rh/1
# =============================================================================


import os
import shutil
from tqdm import tqdm
import random

ACTUAL_PATH = os.getcwd()

if os.name == 'nt':  # Windows
    BASE_PATH = os.path.join(ACTUAL_PATH, "YOLO_train\\ASLYSet\\ASLYset")
    output_path = os.path.join(ACTUAL_PATH, "YOLO_train/datasets")
elif os.name == 'posix':  # Linux
    BASE_PATH = os.path.join(ACTUAL_PATH, "YOLO_train/ASLYSet/ASLYset")
    output_path = os.path.join(ACTUAL_PATH, "YOLO_train/datasets")
else:
    print("OS non supporté.")
    exit(1)

os.makedirs(output_path, exist_ok=True)

# Liste des classes à utiliser
all_classes = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
    "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
]

excluded_classes = ["fn", "sp"]  # Classes à ignorer
classes = [cls for cls in all_classes if cls not in excluded_classes]

class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

# Seed pour reproductibilité
random.seed(42)


def split_data(data, train_ratio=0.9):
    """
    Répartir les données en train et test selon un ratio.
    :param data: nom des images
    :param train_ratio: ratio du dataset de train
    :return: data splitté en train et test selon le train_ratio
    """
    random.shuffle(data)
    split_index = int(len(data) * train_ratio)
    return data[:split_index], data[split_index:]


def process_user_data(user):
    """
    Divise en train test, renomme les images et labels avec un nom unique, corrige le label,
    :param user:
    :return:
    """
    user_images_path = os.path.join(BASE_PATH, "images", user)
    user_labels_path = os.path.join(BASE_PATH, "labels", user)

    # Chemins de sortie pour Train et Test
    train_images_output_path = os.path.join(output_path, "train", "images")
    train_labels_output_path = os.path.join(output_path, "train", "labels")
    test_images_output_path = os.path.join(output_path, "test", "images")
    test_labels_output_path = os.path.join(output_path, "test", "labels")

    os.makedirs(train_images_output_path, exist_ok=True)
    os.makedirs(train_labels_output_path, exist_ok=True)
    os.makedirs(test_images_output_path, exist_ok=True)
    os.makedirs(test_labels_output_path, exist_ok=True)

    for class_name in tqdm(os.listdir(user_images_path),
                           desc=f"Processing {user}"):  # pour chaque classe de la personne
        if class_name not in classes:
            continue  # Ignorer les classes exclues

        class_images_path = os.path.join(user_images_path, class_name)
        class_labels_path = os.path.join(user_labels_path, class_name)

        images = [
            img_name for img_name in os.listdir(class_images_path)
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        train_images, test_images = split_data(images)

        for img_name, split in zip(train_images + test_images,
                                   ['train'] * len(train_images) + ['test'] * len(test_images)):
            # Définir les chemins d'entrée et sortie
            img_path = os.path.join(class_images_path, img_name)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(class_labels_path, label_name)

            new_img_name = f"{split}_{user}_{class_name}_{img_name}".replace(" ", "_")
            new_label_name = os.path.splitext(new_img_name)[0] + ".txt"

            if split == 'train':
                img_output_path = os.path.join(train_images_output_path, new_img_name)
                label_output_path = os.path.join(train_labels_output_path, new_label_name)
            else:  # Test
                img_output_path = os.path.join(test_images_output_path, new_img_name)
                label_output_path = os.path.join(test_labels_output_path, new_label_name)

            shutil.copy(img_path, img_output_path)

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    label_data = f.read().strip()

                with open(label_output_path, 'w') as label_file:
                    updated_labels = []
                    for line in label_data.splitlines():
                        parts = line.split()  # Découpe la ligne par espaces
                        if len(parts) >= 5:  # Assure qu'il y a au moins la classe et 4 coordonnées
                            parts[0] = str(class_to_idx[class_name])  # Remplace l'indice de classe
                            updated_labels.append(" ".join(parts))  # Reconstruis la ligne
                    label_file.write("\n".join(updated_labels) + "\n")


def process_dataset():
    """
    Appelle process_user_data pour chaque personne du dataset
    :return:
    """
    full_path = os.path.join(BASE_PATH, "images")

    if not os.path.exists(full_path):
        os.makedirs(full_path, exist_ok=True)

    users = os.listdir(full_path)
    for user in users:
        user_path = os.path.join(BASE_PATH, "images", user)
        if os.path.isdir(user_path):
            process_user_data(user)


# ====================================================================================================
if __name__ == '__main__':
    process_dataset()

    print("Traitement terminé.")
