# =============================================================================
#   Fichier : script_dataset_yolo_2.py                                                       #
#   Projet : projet-VA                                                        #
#   Auteur : Groupe Gabin Elise Margot Antoine                                #
#   Email : alboulch1@etu.uqac.ca                                             #
#   Numéro étudiant : LEBA27060300                                            #
#   Date : 21 novembre 2024                                                   #
#
#
# pour mettre en forme le dataset
# https://data.mendeley.com/datasets/xs6mvhx6rh/1
# =============================================================================


import os
import shutil
from tqdm import tqdm
import random

base_path = "C:\\Users\\lebou\\OneDrive - yncréa\\000_UQAC\\Cours Automne\\Vision artificielle\\datasets\\ASLYSet"

output_path = "C:\\Users\\lebou\\OneDrive - yncréa\\000_UQAC\\Cours Automne\\Vision artificielle\\datasets\\ASLYSet_formatted"
os.makedirs(output_path, exist_ok=True)

# Liste des classes à utiliser
all_classes = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
    "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"
]

excluded_classes = ["fn", "sp"]  # Classes à ignorer
classes = [cls for cls in all_classes if cls not in excluded_classes]

# Mapping des indices pour YOLO
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

# Seed pour reproductibilité
random.seed(42)

def split_data(data, train_ratio=0.9):
    """Répartir les données en train et test selon un ratio."""
    random.shuffle(data)
    split_index = int(len(data) * train_ratio)
    return data[:split_index], data[split_index:]

def process_user_data(user):
    # Dossiers pour chaque utilisateur
    user_images_path = os.path.join(base_path, "images", user)
    user_labels_path = os.path.join(base_path, "labels", user)

    # Chemins de sortie pour Train et Test
    train_images_output_path = os.path.join(output_path, "train", "images")
    train_labels_output_path = os.path.join(output_path, "train", "labels")
    test_images_output_path = os.path.join(output_path, "test", "images")
    test_labels_output_path = os.path.join(output_path, "test", "labels")

    os.makedirs(train_images_output_path, exist_ok=True)
    os.makedirs(train_labels_output_path, exist_ok=True)
    os.makedirs(test_images_output_path, exist_ok=True)
    os.makedirs(test_labels_output_path, exist_ok=True)

    # Parcourir les classes pour cet utilisateur
    for class_name in tqdm(os.listdir(user_images_path), desc=f"Processing {user}"):
        if class_name not in classes:
            continue  # Ignorer les classes exclues

        class_images_path = os.path.join(user_images_path, class_name)
        class_labels_path = os.path.join(user_labels_path, class_name)

        images = [
            img_name for img_name in os.listdir(class_images_path)
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        train_images, test_images = split_data(images)

        for img_name, split in zip(train_images + test_images, ['train'] * len(train_images) + ['test'] * len(test_images)):
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
    users = os.listdir(os.path.join(base_path, "images"))
    for user in users:
        user_path = os.path.join(base_path, "images", user)
        if os.path.isdir(user_path):
            process_user_data(user)

# ====================================================================================================
# Appeler la fonction pour le traitement du dataset
process_dataset()

print("Traitement terminé.")