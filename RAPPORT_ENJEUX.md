# Rapport Enjeux

## Consignes

Rapport sur les impacts sociaux et environnementaux de votre projet

Pour montrer le recul sur votre projet et son inclusion dans notre environnement et dans votre formation à l'école, nous vous demanderons de produire un petit rapport, sur les enjeux sociaux et/ou environnementaux de votre projet.

Ce rapport, fruit d'une réflexion du groupe entier, devra faire **entre une et deux pages, au format PDF**. Vous pourrez aborder dans ce rapport toute information pertinente sur l'impact direct ou indirect de votre projet sur des aspects sociétaux, artistiques, légaux, environnementaux, économiques, etc. : il peut s'agir de questions liées à la confidentialité des données que vous collectez, d'une réflexion sur l'impact environnemental (positif ou non) des modèles que vous utilisez, les questions de biais ou d'usage afférentes, etc. et son but est de démontrer un recul critique sur les aspects non purement techniques du projet.

## Plan

### Introduction

- Présenter le projet
- Présenter le but de ce rapport

### Aspects Environnementaux

- Consommation énergétique :
    - plus l'image a une grande résolution, plus il y a de calculs
    - volume important de données stockées (et utilise beaucoup de RAM...)
    - utilisation du GPU
    - Deep Learning (pas encore utilisé)
- Pistes pour réduire cet impact :
    - utiliser des images à plus faible résolution lorsque ce n'est pas dérengeant
    - Avantage : comme notre pipeline est modulable, seules les étapes jugées nécessaires par l'utilisateur sont effectuées
- Effets indirects :
    - Peut potentiellement aider à réduire le renouvellement du matériel photographique
    - Effet rebond : avoir accès (gratuitement) à un logiciel de traitement d'image peut inciter à améliorer la qualité de ses images, sans utilité réelle et réfléchie (ça contribue à augmenter la consommation numérique) 

### Aspects Ethiques et Sociétaux

- Open-source : rendre accessible des outils de traitement d'image (gratuitement, transparence, peut être reproduit, modifié, implémentation python de certains algorithmes difficiles à trouver en ligne)
- Risques liés à la manipulation d'images (authenticité ? désinformation ? Est-ce acceptable de ne plus être fidèle à l'image d'origine ? -> journalisme, preuve, sites de tourisme...)
- Peut renforcer le sentiment d'inégalité, car nécessite du matériel performant (bien plus efficace avec un GPU)
- si deep learning : biais des modèles utilisés, pouvant augmenter les stéréotypes (ex: couleur de peau...)

### Aspects Légaux

ex: confidentialité des données collectées
- Que contiennent les images RAW collectées ? (localisation, appareil, paramètres) : risque de fuite d'information

### Aspects Economiques

- Réduire les dépenses pour des logiciels de traitement d'image
- Promotion de l'open-source

### Aspects Artistiques

- LUT pour donner un style personnel à ses images