# Projet 9 - Formation IA Engineer d'OpenClassrooms

## Développez une preuve de concept

### Contexte

Vous essayez d'obtenir un poste de Data Scientist chez "DataSpace", une entreprise qui accompagne ses clients à concevoir et mettre en œuvre des solutions de data science, tant sur des données structurées que sur des problématiques de traitement d’image ou de texte.

Dans le cadre de votre recrutement, le recruteur aimerait que vous réalisiez une veille et que vous identifiez une méthode plus récente pour améliorer la performance d’un modèle.

J'ai choisi de m’inscrire dans une logique d'amélioration du travail réalisé lors du projet 8 de la formation IA Engineer, dans lequel j’ai implémenté la segmentation sémantique sur le dataset Cityscapes à l’aide d’un modèle U-Net, avec plusieurs variantes.


### Notebooks complets et commentés ci-dessous :

https://github.com/CelineLesur/OCR_IAEngineer_Projet9/blob/main/P7_EDA.ipynb

https://github.com/CelineLesur/OCR_IAEngineer_Projet9/blob/main/P7_modele_base.ipynb

https://github.com/CelineLesur/OCR_IAEngineer_Projet7/blob/main/P7_modele_avance.ipynb


### Découpage des dossiers :
📂 /

main.py → Code principal de l’API FastAPI

startup.sh → Code de démarrage d'Azure

requirements.txt → Liste des packages nécessaires

oryx-manifest.toml → Métadonnées sur le déploiement

README.md → Explication du contexte du projet, de la hierarchie des fichiers et des packages utilisés

📂 notebooks/

P7_EDA.ipynb → Analyse exploratoire des données

P7_modele_base.ipynb → Entraînement et évaluation du modèle de base : régression logistique

P7_modele_avance.ipynb → Entraînement et évaluation de modèles de RNN (sans embeddings, avec embeddings GloVe, avec embeddings Fastext) et ModernBERT

📂 test/

test_api.py → Tests unitaires

### Installation

#### Prerequisites

Python 3.11

#### Dependencies

- fastapi - version : 0.115.11
- torch (https://download.pytorch.org/whl/cpu)
- transformers - version : 4.49.0
- azure-storage-blob
- uvicorn
