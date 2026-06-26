# Rapport Enjeux

## Consignes

Rapport sur les impacts sociaux et environnementaux de votre projet

Pour montrer le recul sur votre projet et son inclusion dans notre environnement et dans votre formation à l'école, nous vous demanderons de produire un petit rapport, sur les enjeux sociaux et/ou environnementaux de votre projet.

Ce rapport, fruit d'une réflexion du groupe entier, devra faire **entre une et deux pages, au format PDF**. Vous pourrez aborder dans ce rapport toute information pertinente sur l'impact direct ou indirect de votre projet sur des aspects sociétaux, artistiques, légaux, environnementaux, économiques, etc. : il peut s'agir de questions liées à la confidentialité des données que vous collectez, d'une réflexion sur l'impact environnemental (positif ou non) des modèles que vous utilisez, les questions de biais ou d'usage afférentes, etc. et son but est de démontrer un recul critique sur les aspects non purement techniques du projet.

## Plan

### Introduction
- Présenter le projet
- Présenter le but de ce rapport
- Références

### Aspects Environnementaux

- Consommation énergétique :
    - plus l'image a une grande résolution, plus il y a de calculs
    - volume important de données stockées (et utilise beaucoup de RAM... : on travaille pour la réduire)
    - utilisation du GPU
    - Deep Learning (pas encore utilisé)
- Pistes pour réduire cet impact :
    - utiliser des images à plus faible résolution lorsque ce n'est pas dérengeant
    - Avantage : comme notre pipeline est modulable, seules les étapes jugées nécessaires par l'utilisateur sont effectuées
- Effets indirects :
    - Peut potentiellement aider à réduire le renouvellement du matériel photographique
    - Effet rebond : avoir accès (gratuitement) à un logiciel de traitement d'image peut inciter à améliorer la qualité de ses images, sans utilité réelle et réfléchie (ça contribue à augmenter la consommation numérique) -> alternative moins cher environnementalement parlant que de dire à une IA générative de créer/améliorer une image.

### Aspects Ethiques et Sociétaux

- Open-source : rendre accessible des outils de traitement d'image (gratuitement, transparence, peut être reproduit, modifié, implémentation python de certains algorithmes difficiles à trouver en ligne) [voir article "L’open source, l’armée de l’ombre du logiciel… et de l’Intelligence artificielle" - Le Monde - 05/01/2025 (note: l'article parle de la tech en général, pas que de l'IA)]
- Risques liés à la manipulation d'images (authenticité ? désinformation ? Est-ce acceptable de ne plus être fidèle à l'image d'origine ? -> journalisme, preuve, sites de tourisme...) -> traitement sur toute l'image (pas "ajout" d'un objet, retirer une personne, etc...)
- Peut renforcer le sentiment d'inégalité, car nécessite du matériel performant (bien plus efficace avec un GPU) -> Fracture numérique ? violence structurelle ? [voir Séminaire 4 cours HSS : "Imagining technologies for peace"], mais nous on est open source, c’est mieux + le nôtre peut fonctionner sur CPU (+ plusieurs algos plus ou moins complexes pour certains noeuds)
- si deep learning : biais des modèles utilisés, pouvant augmenter les stéréotypes (ex: couleur de peau...) -> nous : pas de deep learning ou alors purement mathématique/modèle
- regrouper des technologies qui ne sont pas regroupées dans d'autres logiciels
- peut tourner sur n'importe quelle machine (car en python)

### Aspects Légaux

openSource donc ok, tout en local donc pas de fuite (contrairement à un site en ligne qui pourrait récupérer les infos)
ex: confidentialité des données collectées
- Que contiennent les images RAW collectées ? (localisation, appareil, paramètres) : risque de fuite d'information

### Aspects Economiques

- Réduire les dépenses pour des logiciels de traitement d'image
- Promotion de l'open-source

### Aspects Artistiques

- LUT pour donner un style personnel à ses images