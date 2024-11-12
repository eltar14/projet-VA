# =============================================================================
#   Fichier : qt_app.py                                                       #
#   Projet : projet-VA                                                        #
#   Auteur : Groupe Gabin Elise Margot Antoine                                #
#   Email : alboulch1@etu.uqac.ca                                             #
#   Numéro étudiant : LEBA27060300                                            #
#   Date : 12 novembre 2024                                                   #
#
#
# Fichier contenant l'appli Qt pour l'affichage des prédictions de langue
# des signes ASL.
# =============================================================================

from qtpy.QtWidgets import QApplication, QMainWindow, QLabel, QTextEdit, QVBoxLayout, QWidget, QPushButton, QCheckBox
from qtpy.QtCore import QTimer
from qtpy.QtGui import QImage, QPixmap
import cv2
import sys
from ultralytics import YOLO

class YOLOQtApp(QMainWindow):
    def __init__(self, model, video_path=None):
        super().__init__()
        self.model = model
        self.cap = cv2.VideoCapture(video_path or 0)
        self.is_paused = False  # bool de pause
        self.concat_enabled = False  # bool concaténation des prédictions
        self.detected_text = ""  # texte concaténé
        self.last_detection = None
        self.repetition_counter = 0  # compteur de repetition
        self.repetition_threshold = 25  # seuil de frames consécutives apres lequel on repete la lettre

        # Configurer l'interface de la fenêtre
        self.setWindowTitle("YOLO ASL Detection")
        self.setGeometry(100, 100, 800, 600)

        # Widgets pour l'affichage
        self.video_label = QLabel(self)
        self.text_display = QTextEdit(self) # on pourra régler la taille et la police si on veut.
        self.text_display.setReadOnly(True)  # affichage en lecture seule pour les prédictions
        self.pause_button = QPushButton("Pause", self)  # bouton de pause
        self.concat_checkbox = QCheckBox("Concaténer les prédictions", self)  # Bouton à cocher pour la concaténation

        # Connect
        self.pause_button.clicked.connect(self.toggle_pause)
        self.concat_checkbox.stateChanged.connect(self.toggle_concat)

        # Layout principal
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.text_display)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.concat_checkbox)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Set le temps entre deux frames affichees (donc entre 2 predictions) (constaté que une prédiction prend moins de 15 ms à calculer)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30 ms pour environ 30 fps (1/30 ~= 0.03333), laisse le temps pour prédire

    def toggle_pause(self):
        """
        toggle la mise en pause ou non de la détection des lettres
        :return:
        """
        if self.is_paused: # si en pause et cliqué
            self.timer.start(30)  # Reprendre le timer
            self.pause_button.setText("Pause")
        else:
            self.timer.stop()  # sinon arreter le timer pour mettre en pause
            self.pause_button.setText("Reprendre")
        self.is_paused = not self.is_paused

    def toggle_concat(self, state):
        """
        toggle de la concatenation des classes des détections pour former un mot
        :param state:
        :return:
        """
        self.concat_enabled = bool(state)
        if not self.concat_enabled:
            self.detected_text = ""  # RAZ si on arrête la concaténation

    def update_frame(self):
        """
        analyse d'une image avec YOLO et update des affichages + case de texte
        :return: rien, affichage.
        """
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            print("Erreur lors de la lecture du flux vidéo.")
            return

        results = self.model(frame) # faire une prédiction sur la frame
        result_frame = results[0].plot()  # ajouter les boxes et les labels sur la frame

        # Extraction des labels des détectiobs
        current_detection = [] # liste des current frame detectionS


        # Concatener ou non les prédictions
        if self.concat_enabled:
            for detection in results[0].boxes:  # pour chaque elt detecte
                class_id = int(detection.cls[0])
                label = self.model.names[class_id]  # on trouve son label

                # Vérifie si la détection actuelle est la même que la dernière
                if label == self.last_detection:
                    self.repetition_counter += 1
                else:
                    self.repetition_counter = 0  # réinitialiser si la détection change

                # Ajoute la lettre seulement si elle est différente de la dernière
                # OU si le seuil de répétition est atteint
                if label != self.last_detection or self.repetition_counter >= self.repetition_threshold:
                    current_detection.append(label)
                    self.last_detection = label
                    self.repetition_counter = 0  # Réinitialiser après ajout
                    break


            self.detected_text += " ".join(current_detection) + " "
        else:

            for detection in results[0].boxes:
                class_id = int(detection.cls[0])
                label = self.model.names[class_id]
                current_detection.append(label)
            self.detected_text = "\n".join(current_detection)

        # Mise à jour de l'affichage avec le texte avec les signes/objets détectés
        self.text_display.setText(self.detected_text)

        # Conversion RGB pour affichage
        rgb_image = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image)) # affichage

    def closeEvent(self, event):
        """
        libérer tout à la fin
        :param event:
        :return:
        """
        # Arreter le flux vidéo quand la fenêtre se ferme
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

def run_yolo_app(model, video_path=None):
    """
    Prend en argument le modèle et en option le chemin d'une vidéo (si vide il prend la webcam par défaut de l'ordi),
    à la même manière que demo dans basic_input_test.py.
    :param model: modèle YOLO
    :param video_path: chemin d'une vidéo (MP4, ...) ou vide.
    :return: rien, affichage de l'appli Qt
    """
    app = QApplication(sys.argv)
    main_window = YOLOQtApp(model, video_path)
    main_window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")  # Modèle YOLO pré-entraîné pour tester
    run_yolo_app(model, video_path="videos/video1.mp4")  # ou laisser video_path vide ou None pour utiliser la webcam
