# =============================================================================
#   Fichier : basic_input_test.py                                             #
#   Projet : projet-VA                                                        #
#   Auteur : Groupe Gabin Elise Margot Antoine                                #
#   Date : 12 novembre 2024                                                   #
#
#
# Fichier contenant une mini demo d'affichage de prédiction yolo.
# =============================================================================
import cv2
from ultralytics import YOLO


def infere(model, cap):
    """
    Prend en argument un modèle YOLO et une capture (cap)
    :param model: modèle YOLO
    :param cap: capture CV2
    :return: rien, juste l'affichage
    """
    if not cap.isOpened():
        print("Erreur lors de l'ouverture.")
        exit()
    while cap.isOpened():
        # for each frame
        ret, frame = cap.read()
        if not ret:
            print("No frames left.")
            break
        # Exécuter l'inférence sur l'image
        results = model(frame)  # Retourne un objet Results
        # Afficher les résultats sur l'image
        result_frame = results[0].plot()
        # Afficher l'image avec les détections
        cv2.imshow("YOLO", result_frame)
        # Quitter si 'q' est pressé
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # free resources
    cap.release()
    cv2.destroyAllWindows()


def demo(model, video_path=None):
    """
    Fonction prenant en argument un modèle YOLO affichant les prédictions. On peut remplir l'argument video_path
    pour faire des prédictions à partir d'un fichier video existant (type MP4). Si laissé vide, il utilisera la
    webcam par défaut de l'ordi.
    :param model: modele YOLO
    :param video_path: chemin ou None,
    :return: rien, affichage CV2 seulement
    """
    if video_path is not None:
        cap = cv2.VideoCapture(video_path)  # depuis un fichier video
    else:
        cap = cv2.VideoCapture(0)  # depuis la webcam principale

    infere(model, cap)


if __name__ == '__main__':
    #demo(YOLO("yolo11n.pt"), "videos/video1.mp4") # https://www.youtube.com/watch?v=bwJ-TNu0hGM par exemple
    demo(YOLO("yolo11n.pt"))
