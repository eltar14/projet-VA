# 8INF804 - Projet de session

## Introduction

L'accessibilité est une priorité croissante dans le développement technologique, visant à créer des outils qui améliorent la vie des personnes en situation de handicap. Le premier TP s'inscrivait dans ce cadre, et a motivé le choix de ce sujet pour ce projet, dans un contexte différent, celui de la langue des signes, utilisée par les malentendants pour communiquer. Notre objectif est donc ici de développer un *Proof of Concept* de la faisabilité d'une application de reconnaissance de la langue des signes en temps réel, basée sur des modèles de détection d'objets.

Cette application repose sur un modèle YOLO (You Only Look Once), entraîné spécifiquement sur un *dataset* de lettres et de chiffres en langue des signes. Grâce à ce modèle de détection d'objets, notre solution est capable de déchiffrer des signes en temps réel via des flux vidéo de *webcam* ou des vidéos préenregistrées. YOLO, en raison de sa rapidité et de son efficacité pour détecter et classer des objets, est parfaitement adapté pour une reconnaissance en temps réel, essentielle pour interpréter les signes de manière fluide et précise.

L'objectif principal de cette application est de démontrer la faisabilité de l'accessibilité et l'aide aux personnes malentendantes pour communiquer plus facilement avec leur entourage, en particulier dans des situations où un traducteur de langue des signes n'est pas disponible. Le *scope* de cette application s'inscrit dans le cadre d'un démonstrateur sur des signes simples (lettres et chiffres), en ignorant l'ensemble du dictionnaire de langue des signes, pour permettre au projet de conserver la dimension d'un projet de fin de session. 

## Implémentation

### Modèle

Le modèle YOLOv3 a été choisi pour sa rapidité et son efficacité. Il est capable de détecter et de classer des objets en temps réel, ce qui est essentiel pour notre application. Nous avons entraîné ce modèle sur un *dataset* de lettres et de chiffres en langue des signes, pour qu'il soit capable de reconnaître ces signes en temps réel.

### *Dataset*

> TODO : Le dataset envoyé sur whatsapp ne comporte que des lettres, je laisse ça vide pour l'instant, à compléter

YOLO nécessite, pour son entraînement, que les objets à détecter soient idéalement entourés de *bounding boxes*, ou à défauts situés au centre de l'image. Nous avons donc choisi le...

### Méthodologie

> TODO : Méthodologie de l'entraînement du modèle, à compléter

### Choix de conception

> TODO : Choix de conception de l'application, à compléter

## Résultats

> TODO : Résultats de l'entraînement du modèle, à compléter

## Conclusion

> TODO : Conclusion si tout se passe bien, à modifier sinon

Dans ce projet nous avons réussis à mettre en place un modèle de détection d'objets capable de reconnaître des signes en langue des signes en temps réel. Nous avons entraîné ce modèle sur un *dataset* de lettres et de chiffres en langue des signes, et avons obtenu des résultats satisfaisants. Notre application est capable de détecter et de classer des signes en temps réel, et peut être utilisée via une *webcam* ou sur des vidéos préenregistrées.

Cette application avait une visée de démonstrateur, et se borne donc à reconnaître des lettres et chiffres. Dans un contexte réel, une discussion en langue des signes implique de signer de nombreux mots, à peu près autant qu'en langue vocale, et nécessiterait un modèle beaucoup plus complexe, capable de reconnaître un grand nombre de signes différents, avec un temps d'entraînement d'autant conséquent. Ce serait la piste d'amélioration la plus évidente pour ce projet, en étendant le *dataset* et en entraînant le modèle sur un plus grand nombre de signes.