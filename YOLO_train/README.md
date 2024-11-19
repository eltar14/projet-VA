# README YOLO Training

Pour réaliser un entraînement il faut :

- Télécharger le [dataset de Kaggle](https://www.kaggle.com/datasets/daskoushik/sign-language-dataset-for-yolov7)
- Le décompresser et le mettre dans un dossier `datasets` (nom défini par YOLO, à respecter !)

Packages : 

```bash
pip install ultralytics
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Ça ne marche pas avec un `requirements.txt`, il faut installer les packages un par un.


Il y a 2 possibilités pour entrainer le modèle :

- Soit en utilisant le script `yolo_train.py` (plus simple à lancer depuis un *VEnv*)
- Soit avec le *notebook* `training.ipynb` (plus simple pour visualiser les résultats, mais peut nécessiter la création d'un *kernel* avec les dépendances du VEnv)

Les 2 méthodes sont vraiment similaires, le *notebook* est juste un peu plus complet pour visualiser les résultats.

  
Interpréter les résultats :

- Les résultats sont stockés dans le dossier `runs/detect/trainX`
- Le dossier `weights\ contient les poids du modèle 
- Il y a des images des *batch* de `train` et `val` pour voir un peu ce que retrouve le modèle
- Il y a également 2 matrices de confusions (une normalisée et l'autre non)
- Enfin, les courbes, globalement on les veut le plus haut possible 

J'ai mis `runs` dans le gitignore parce que ça fait beaucoup de fichiers, je vais juste *push* un exemple de *run* pour que vous puissiez voir ce que ça donne, et on mettra également à jour lorsqu'on aura notre modèle final.