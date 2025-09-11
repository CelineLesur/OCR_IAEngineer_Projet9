import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import os
import base64
from io import BytesIO
import numpy as np

def get_img_base64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def build_multiindex_from_colnames(df):
    cols = df.columns.tolist()
    tuples = []
    for c in cols:
        # on sépare sur le dernier espace pour garder les modèles pouvant contenir des espaces
        if "-" in c:
            model, metric = c.rsplit("-", 1)
        else:
            # fallback si pas de séparateur
            model, metric = c, ""
        tuples.append((model, metric))
    # Sanity check : même taille
    if len(tuples) != len(cols):
        raise ValueError(f"Mismatch columns ({len(cols)}) vs tuples ({len(tuples)})")
    return pd.MultiIndex.from_tuples(tuples, names=["Modèle", "Métrique"])

# STREAMLIT CONFIG
st.set_page_config(
    page_title="Dashboard",
    layout="wide"
)

# Fonction principale
def main():
    st.title("Comparaisons de modèles pour la segmentation Sémantique")

    # --- Création d'une selectbox avec session_state ---
    if "selected_image" not in st.session_state:
        st.session_state.selected_image = "frankfurt_000000_000294_leftImg8bit.png"

    def sync_from_tab2():
        st.session_state.selected_image = st.session_state.tab2_selection

    def sync_from_tab3():
        st.session_state.selected_image = st.session_state.tab3_selection

    def sync_from_tab4():
        st.session_state.selected_image = st.session_state.tab4_selection

    def sync_from_tab5():
        st.session_state.selected_image = st.session_state.tab5_selection

    # Création de 3 onglets
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["EDA", "U-Net", "SegFormer", "Mask2Former", "Synthèse"])

     # --------------------- EDA ---------------------
    with tab1:
        st.header("Jeu de données Cityscapes")

    # Texte explicatif sur Cityscapes
        st.markdown("""
        Cityscapes est un jeu de données de référence dans le domaine de la segmentation sémantique urbaine.
        Il se compose de 5 000 images RGB haute résolution (2048×1024) prises dans 50 villes allemandes depuis des caméras embarquées, couvrant des scènes variées.
        Chaque pixel est annoté et associé à une classe sémantique de 0 à 33 représentant un élément de la scène (route, trottoir, voiture, piéton, etc.).
        """)
        st.markdown("""
        Voici un exemple d'image de Cityscapes avec son masque de segmentation associé :
        """)
        img1 = Image.open("images/cityscapes_1.jpg")
        col1,col2,col3 = st.columns([1,2,1])
        with col2:
            st.image(img1, use_column_width=True)
        # img1_base64 = get_img_base64(img1)

        # st.markdown(
        #     f"""
        #     <div style="text-align: center;">
        #         <img src="data:image/png;base64,{img1_base64}" alt="image" style="max-width:100%; height:auto;">
        #         <p style="font-size:14px; color:gray;">Exemple d’une image Cityscapes et son masque associé</p>
        #     </div>
        #     """,
        #     unsafe_allow_html=True
        # )

        st.markdown("""
        Dans ce projet, nous avons regroupé les 34 classes initiales en 8 catégories : vide, route/trottoir, construction, objet, nature, ciel, humain et vehicule.
        """)
        st.markdown("""
        Regardons la distribution de ces 8 catégories dans les images du jeu d'entraînement (2975 images) et de validation (500 images) :
        """)
        img2 = Image.open("images/cityscapes_2.jpg")
        img2_base64 = get_img_base64(img2)

        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{img2_base64}" alt="image" style="max-width:35%; height:auto;">
                <p style="font-size:14px; color:gray;">Distribution des 8 catégories dans le jeu d'entraînement et de validation</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("""
        On peut constater un fort déséquilibre des classes dans le jeu de données Cityscapes. Ce sera une limitation pour notre projet de segmentation sémantique car il sera plus difficile pour les modèles de prédire les classes rares.
        """)
    
    # with tab2:
    #     st.header("Prédictions avec U-Net")

    #     # --------------------- UNET ---------------------
    #     st.markdown("""
    #     Le meilleur modèle U-Net a été entraîné sur 50 epochs à partir d'images RGB de résolutions 224x224. Voici les hyper-paramètres  de ce modèle :
    #     """)
    #     st.markdown("""
    #     - avec encodeur pré-entraîné VGG16,
    #     """)
    #     st.markdown("""
    #     - avec légère data augmentation,
    #     """)
    #     st.markdown("""
    #     - avec trois couches de filtres [64,128,256],
    #     """)
    #     st.markdown("""
    #     - avec l'optimiseur Adam,
    #     """)
    #     st.markdown("""
    #     - avec la fonction de perte focal dice,
    #     """)
    #     st.markdown("""
    #     - avec un taux de drop out de 0.3,
    #     """)
    #     st.markdown("""
    #     - et avec un taux d'apprentissage de 0.0001.
    #     """)
        
    #     # Choix de l'image
    #     IMG_DIR = "images"  
    #     # Charger les fichiers d'images du dossier
    #     image_files = [f for f in os.listdir(IMG_DIR) if not f.startswith("pred") and f.lower().endswith((".png"))]

    #     if not image_files :
    #         st.warning("Aucune image trouvée")
    #         return

    #     st.selectbox(
    #     "Choisissez une image :",
    #     image_files,
    #     key="tab2_selection",
    #     index=image_files.index(st.session_state.selected_image),
    #     on_change=sync_from_tab2
    #     )

    #     if st.session_state.selected_image  :
    #         img_path = os.path.join(IMG_DIR, f"pred_unet_{st.session_state.selected_image }")
    #         image = Image.open(img_path)
    #         target_height = 300
    #         w, h = image.size
    #         new_w = int(w * (target_height / h))
    #         img_resized = image.resize((new_w, target_height))
    #         st.image(img_resized)

    #     st.markdown("""
    #     Visuellement, nous observons une très bonne performance globale sur les classes fréquentes (ciel, route, véhicule, nature), moindre sur les classes rares (objet, humain) qui restent mal segmentées, confondues avec d’autres classes en arrière plan.
    #     """)     

    #     st.header("Résultat de l'entraînement")
    #     st.markdown("""
    #     Voici les courbes d'entraînement de ce modèle :
    #     """)
    #     img3 = Image.open("images/unet_1.jpg")
    #     img3_base64 = get_img_base64(img3)

    #     st.markdown(
    #         f"""
    #         <div style="text-align: center;">
    #             <img src="data:image/png;base64,{img3_base64}" alt="image" style="max-width:60%; height:auto;">
    #             <p style="font-size:14px; color:gray;">Courbes d'entraînement sur 50 epochs du modèle U-Net défini précédemment</p>
    #         </div>
    #         """,
    #         unsafe_allow_html=True
    #     )
    #     st.markdown("""
    #     On voit une croissance rapide de l'accuracy et de l'IoU donc le modèle apprend rapidement les classes dominantes et on observe une converge après une dizaine d'epochs. L'écart entre les courbes d'entrainement et de validation est faible ce qui prouve qu'il n'y a pas de surappentissage marqué.
    #     """)  

    #     # Résultats globaux 
    #     global_metrics = ["IoU global (%)", "IoU pondéré (%)", "IoU macro (%)", "Dice global (%)", "Dice pondéré (%)", "Dice macro (%)"]
    #     global_values =  [69.9, 82.1, 71.8, 88.9, 89.6, 82.0]
    #     table_data = [[m, f"{v:.1f}"] for m, v in zip(global_metrics, global_values)]
    
    #     # Résultats par classe
    #     class_metrics = {
    #         "Classe": ["Vide", "Route/Trottoir", "Construction/Bâtiment", "Objet", "Nature", "Ciel", "Humain", "Véhicule"],
    #         "IoU (%)": [66.0, 91.2, 73.0, 21.9, 79.8, 86.1, 41.7, 73.8],
    #         "Dice (%)": [79.6, 95.4, 84.4, 36.0, 88.8, 92.6, 58.8, 85.0]
    #     }
    #     df_classes = pd.DataFrame(class_metrics)

    #     col1,col2 = st.columns(2)
    #     with col1 :
    #         st.subheader("Métriques globales de U-Net")
    #         # Créer la figure
    #         fig1, ax1 = plt.subplots(figsize=(6,3))
    #         ax1.axis('off')  # Masquer les axes

    #         # Créer la table
    #         table = ax1.table(
    #             cellText=table_data,
    #             colLabels=["Métrique", "Valeur"],
    #             cellLoc='center',
    #             loc='center')

    #         table.scale(1, 1.5)  # Ajuster la taille
            
    #         # Forcer le fond blanc
    #         fig1.patch.set_facecolor('white')
    #         ax1.set_facecolor('white')

    #         # Afficher dans Streamlit
    #         st.pyplot(fig1)

    #     with col2 :
    #         st.subheader("Métriques par classe de U-Net")
    #         fig2, ax2 = plt.subplots(figsize=(6,4))
            
    #         # --- Heatmap ---
    #         sns.heatmap(
    #             df_classes.set_index("Classe"),
    #             annot=True, fmt=".1f", cmap="YlGnBu", cbar=False, ax=ax2
    #         )
    #         st.pyplot(fig2)

    #     st.markdown("""
    #     U-Net est efficace sur les grandes classes majoritaires, ce qui confirme sa pertinence pour une segmentation de scènes routières. En revanche, le modèle échoue sur les petites classes fines (piétons, poteaux), probablement à cause de la taille réduite des objets (difficiles à capter dans l’architecture en downsampling/upsampling) et le déséquilibre du dataset (ces classes sont sous-représentées).
    #     """) 

    # with tab3:
    #     st.header("Prédictions avec SegFormer B1")

    #     # --------------------- SEGFORMER ---------------------
    #     st.markdown("""
    #     Le meilleur modèle SegFormer a été entraîné sur 50 epochs à partir d'images RGB de résolutions 512x1024. Voici les hyper-paramètres  de ce modèle :
    #     """)
    #     st.markdown("""
    #     - sans encodeur pré-entraîné,
    #     """)
    #     st.markdown("""
    #     - avec légère data augmentation,
    #     """)
    #     st.markdown("""
    #     - avec 4 blocs Transformers de deux couches et les dimensions cachées correspondantes [64, 128, 320, 512],
    #     """)
    #     st.markdown("""
    #     - avec l'optimiseur Adam,
    #     """)
    #     st.markdown("""
    #     - avec la fonction de perte cross entropy,
    #     """)
    #     st.markdown("""
    #     - avec une taille de batch de 8,
    #     """)
    #     st.markdown("""
    #     - avec une décroissance des poids (weight decay) de 0.01,
    #     """)
    #     st.markdown("""
    #     - et avec un taux d'apprentissage de 0.0005.
    #     """)

    #     # Liste déroulante
    #     st.selectbox(
    #     "Choisissez une image :",
    #     image_files,
    #     key="tab3_selection",
    #     index=image_files.index(st.session_state.selected_image),
    #     on_change=sync_from_tab3
    #     )

    #     if st.session_state.selected_image  :
    #         img_path = os.path.join(IMG_DIR, f"pred_segformer_{st.session_state.selected_image }")
    #         image = Image.open(img_path)
    #         target_height = 300
    #         w, h = image.size
    #         new_w = int(w * (target_height / h))
    #         img_resized = image.resize((new_w, target_height))
    #         st.image(img_resized)

    #     st.markdown("""
    #     Visuellement, SegFormer-B1 donne des résultats corrects pour les grandes classes structurantes (route, voiture, bâtiment, ciel, nature), mais il montre ses limites pour les petits objets fins (poteaux, panneaux), les zones de transition et la précision des contours. C’est typique d’un modèle léger et entraîné from scratch : il capte les classes fréquentes mais échoue sur les détails.        
    #     """)

    #     st.header("Résultat de l'entraînement")
    #     st.markdown("""
    #     Voici les courbes d'entraînement de ce modèle :
    #     """)
    #     img4 = Image.open("images/segformer_1.jpg")
    #     img4_base64 = get_img_base64(img4)
    #     st.markdown(
    #         f"""
    #         <div style="text-align: center;">
    #             <img src="data:image/png;base64,{img4_base64}" alt="image" style="max-width:75%; height:auto;">
    #             <p style="font-size:14px; color:gray;">Courbes d'entraînement sur 50 epochs du modèle SegFormer défini précédemment</p>
    #         </div>
    #         """,
    #         unsafe_allow_html=True
    #     )

    #     st.markdown("""
    #     SegFormer-B1 converge bien et reste stable sans surapprentissage. Les résultats montrent qu’il généralise correctement mais son IoU plafonne aux alentours de 0.6 ce qui montre une moindre performance. 
    #     """)

    #     # Résultats globaux 
    #     global_metrics_seg = ["IoU global (%)", "IoU pondéré (%)", "IoU macro (%)", "Dice global (%)", "Dice pondéré (%)", "Dice macro (%)"]
    #     global_values_seg =  [65.4, 78.6, 65.8, 87.8, 87.2, 76.7]
    #     table_data_seg = [[m, f"{v:.1f}"] for m, v in zip(global_metrics_seg, global_values_seg)]
    
    #     # Résultats par classe
    #     class_metrics_seg = {
    #         "Classe": ["Vide", "Route/Trottoir", "Construction/Bâtiment", "Objet", "Nature", "Ciel", "Humain", "Véhicule"],
    #         "IoU (%)": [56.7, 90.1, 76.6, 18.2, 81.0, 80.4, 46.3, 77.3],
    #         "Dice (%)": [72.4, 94.8, 86.7, 30.9, 89.5, 89.1, 63.3, 87.2]
    #     }
    #     df_classes_seg = pd.DataFrame(class_metrics_seg)

    #     col1,col2 = st.columns(2)
    #     with col1 :
    #         st.subheader("Métriques globales de SegFormer")
    #         # Créer la figure
    #         fig1, ax1 = plt.subplots(figsize=(6,3))
    #         ax1.axis('off')  # Masquer les axes

    #         # Créer la table
    #         table = ax1.table(
    #             cellText=table_data_seg,
    #             colLabels=["Métrique", "Valeur"],
    #             cellLoc='center',
    #             loc='center')

    #         table.scale(1, 1.5)  # Ajuster la taille
            
    #         # Forcer le fond blanc
    #         fig1.patch.set_facecolor('white')
    #         ax1.set_facecolor('white')

    #         # Afficher dans Streamlit
    #         st.pyplot(fig1)

    #     with col2 :
    #         st.subheader("Métriques par classe de SegFormer")
    #         fig2, ax2 = plt.subplots(figsize=(6,4))
            
    #         # --- Heatmap ---
    #         sns.heatmap(
    #             df_classes_seg.set_index("Classe"),
    #             annot=True, fmt=".1f", cmap="YlGnBu", cbar=False, ax=ax2
    #         )
    #         st.pyplot(fig2)

    #     st.markdown("""
    #     On observe une bonne performance globale. Les métriques pondérées sont élevées, ce qui montre que les classes fréquentes sont bien apprises. SegFormer excelle donc sur les grandes classes bien définies comme la route, la nature, le ciel mais il a plus de mal sur les petites classes ou celles avec beaucoup de variation comme les objets ou les humains. Cela confirme que l’architecture B1 légère et from scratch est limitée, surtout sur un dataset de taille réduite. Un backbone plus puissant (B3, B4, B5) ou un encodeur pré-entraîné transformerait probablement ces résultats.
    #     """) 

    # with tab4:
    #     st.header("Prédictions avec Mask2Former")

    #     # --------------------- MASK2FORMER ---------------------
    #     st.markdown("""
    #     Le modèle Mask2Former pré-entraîné sur Cityscapes a été utilisé en inférence puisque seuls des modèles pré-entraînés sont disponibles. Voici les hyper-paramètres  de ce modèle :
    #     """)
    #     st.markdown("""
    #     - avec encodeur Swin-Transformer pré-entraîné sur Cityscapes,
    #     """)
    #     st.markdown("""
    #     - avec une data augmentation (RandomResize, RandomCrop, RandomFlip),
    #     """)
    #     st.markdown("""
    #     - avec 10 couches Transformers, 100 requêtes et 8 têtes d'attention,
    #     """)
    #     st.markdown("""
    #     - avec l'optimiseur AdamW,
    #     """)
    #     st.markdown("""
    #     - avec une taille de batch de 16,
    #     """)
    #     st.markdown("""
    #     - avec une décroissance des poids (weight decay) de 0.05,
    #     """)
    #     st.markdown("""
    #     - et avec un taux d'apprentissage de 0.0001.
    #     """)

    #     # Liste déroulante
    #     st.selectbox(
    #     "Choisissez une image :",
    #     image_files,
    #     key="tab4_selection",
    #     index=image_files.index(st.session_state.selected_image),
    #     on_change=sync_from_tab4
    #     )

    #     if st.session_state.selected_image  :
    #         img_path = os.path.join(IMG_DIR, f"pred_mask2former_{st.session_state.selected_image }")
    #         image = Image.open(img_path)
    #         target_height = 300
    #         w, h = image.size
    #         new_w = int(w * (target_height / h))
    #         img_resized = image.resize((new_w, target_height))
    #         st.image(img_resized)

    #     st.markdown("""
    #     Visuellement, on observe une excellente robustesse sur les grandes classes (route, bâtiment, ciel, voitures), les contours sont nets et les ombres sont bien gérés par le modèle. Des erreurs perdurent sur les objets fins et rares (poteaux, feux, panneaux) mais moins qu'avec les autres modèles.
    #     """)

    #     st.header("Résultat de l'inférence")

    #     st.markdown("""
    #     Comme nous avons utilisé Mask2Former seulement en inférence, nous n'avons pas les courbes d'entraînement. Cependant, voici les résultats du modèle :
    #     """) 

    #     # Résultats globaux 
    #     global_metrics_mask = ["IoU pondéré (%)", "IoU macro (%)", "Dice global (%)", "Dice pondéré (%)", "Dice macro (%)"]
    #     global_values_mask =  [91.0, 70.8, 95.1, 95.0, 77.4]
    #     table_data_mask = [[m, f"{v:.1f}"] for m, v in zip(global_metrics_mask, global_values_mask)]
    
    #     # Résultats par classe
    #     class_metrics_mask = {
    #         "Classe": ["Vide", "Route/Trottoir", "Construction/Bâtiment", "Objet", "Nature", "Ciel", "Humain", "Véhicule"],
    #         "IoU (%)": [00.0, 98.8, 93.6, 65.0, 93.7, 95.8, 78.3, 93.4],
    #         "Dice (%)": [00.0, 97.7, 87.9, 48.2, 88.2, 91.9, 64.3, 87.6]
    #     }
    #     df_classes_mask = pd.DataFrame(class_metrics_mask)

    #     col1,col2 = st.columns(2)
    #     with col1 :
    #         st.subheader("Métriques globales de Mask2Former")
    #         # Créer la figure
    #         fig1, ax1 = plt.subplots(figsize=(6,3))
    #         ax1.axis('off')  # Masquer les axes

    #         # Créer la table
    #         table = ax1.table(
    #             cellText=table_data_mask,
    #             colLabels=["Métrique", "Valeur"],
    #             cellLoc='center',
    #             loc='center')

    #         table.scale(1, 1.5)  # Ajuster la taille
            
    #         # Forcer le fond blanc
    #         fig1.patch.set_facecolor('white')
    #         ax1.set_facecolor('white')

    #         # Afficher dans Streamlit
    #         st.pyplot(fig1)

    #     with col2 :
    #         st.subheader("Métriques par classe de Mask2Former")
    #         fig2, ax2 = plt.subplots(figsize=(6,4))
            
    #         # --- Heatmap ---
    #         sns.heatmap(
    #             df_classes_mask.set_index("Classe"),
    #             annot=True, fmt=".1f", cmap="YlGnBu", cbar=False, ax=ax2
    #         )
    #         st.pyplot(fig2)

    #     st.markdown("""
    #     Mask2Former atteint une excellente performance globale (IoU pondéré 91 %, Dice global 95 %), bien supérieure aux approches classiques même s'il est difficile de faire la comparaison puisque Mask2Former ne prédit pas la classe "vide". Les classes rares (humains, objets) sont encore difficiles à segmenter c'est pourquoi les métriques macro sont plus faibles. Ces résultats confirment que Mask2Former est un modèle très adapté à la segmentation de scènes urbaines, mais qui bénéficierait de stratégies plus évoluées pour rééquilibrer les classes rares.
    #     """) 

    # with tab5:
    #     st.header("Synthèse des résultats")

    #     # Liste déroulante
    #     st.selectbox(
    #     "Choisissez une image :",
    #     image_files,
    #     key="tab5_selection",
    #     index=image_files.index(st.session_state.selected_image),
    #     on_change=sync_from_tab5
    #     )

    #     if st.session_state.selected_image  :
    #         img_path = os.path.join(IMG_DIR, f"pred_synthese_{st.session_state.selected_image }")
    #         image = Image.open(img_path)
    #         image_base64 = get_img_base64(image)
    #         st.markdown(
    #             f"""
    #             <div style="text-align: center;">
    #                 <img src="data:image/png;base64,{image_base64}" alt="image" style="max-width:100%; height:auto;">
    #                 <p style="font-size:14px; color:gray;">Exemple d’une image Cityscapes et son masque associé</p>
    #             </div>
    #             """,
    #             unsafe_allow_html=True
    #         )

    #     st.markdown("""
    #     Visuellement, on voit qu'avec U-Net les grandes classes sont globalement détectées mais on aperçoit du bruit avec des pixels mal classés et les classes minoritaires son mal représentées. Avec SegFormer B1, il y a une meilleure capture de la structure globale donc il prédit  de surfaces plus uniformes sans bruits. Cependant, les détails fins, comme les poteaux ou de façon plus général les contours, ne sont pas forcément bien respectés. Finalement, Mask2Former parvient lui à prédire un masque proche du masque réel avec une très bonne segmentation des grandes classes et une meilleure réprésentation des objets fin que les deux autres modèles, même si ce n'est pas encore parfait.
    #     """) 

    #     # Résultats globaux 
    #     global_metrics_synt = ["IoU global (%)", "IoU pondéré (%)", "IoU macro (%)", "Dice global (%)", "Dice pondéré (%)", "Dice macro (%)"]
    #     global_values_synt_unet =  [69.9, 82.1, 71.8, 88.9, 89.6, 82.0]
    #     global_values_synt_seg =  [65.4, 78.6, 65.8, 87.8, 87.2, 76.7]
    #     global_values_synt_mask =  ["-",91.0, 70.8, 95.1, 95.0, 77.4]
    #     table_data_synt = [[t, f"{u:.1f}", f"{s:.1f}", m] for t, u, s, m in zip(global_metrics_synt, global_values_synt_unet,global_values_synt_seg,global_values_synt_mask)]
    
    #     # Résultats par classe
    #     class_metrics_synt = {
    #     "Classe": ["Vide", "Route/Trottoir", "Construction/Bâtiment", "Objet", "Nature", "Ciel", "Humain", "Véhicule"],
    #     "UNet-IoU": [66.0, 91.2, 73.0, 21.9, 79.8, 86.1, 41.7, 73.8],
    #     "UNet-Dice": [79.6, 95.4, 84.4, 36.0, 88.8, 92.6, 58.8, 85.0],
    #     "SegFormer-IoU": [56.7, 90.1, 76.6, 18.2, 81.0, 80.4, 46.3, 77.3],
    #     "SegFormer-Dice": [72.4, 94.8, 86.7, 30.9, 89.5, 89.1, 63.3, 87.2],
    #     "Mask2Former-IoU": [00.0, 98.8, 93.6, 65.0, 93.7, 95.8, 78.3, 93.4],
    #     "Mask2Former-Dice": [00.0, 97.7, 87.9, 48.2, 88.2, 91.9, 64.3, 87.6],
    #     }
    #     df_classes_synt = pd.DataFrame(class_metrics_synt)

    #     col1,col2 = st.columns(2)
    #     with col1 :
    #         st.subheader("Comparatif des métriques globales")
    #         # Créer la figure
    #         fig1, ax1 = plt.subplots(figsize=(6,3))
    #         ax1.axis('off')  # Masquer les axes

    #         # Créer la table
    #         table = ax1.table(
    #             cellText=table_data_synt,
    #             colLabels=["Métrique", "U-Net", "SegFormer", "Mask2Former"],
    #             cellLoc='center',
    #             loc='center')

    #         table.scale(1, 1.5)  # Ajuster la taille
            
    #         # Forcer le fond blanc
    #         fig1.patch.set_facecolor('white')
    #         ax1.set_facecolor('white')

    #         # Afficher dans Streamlit
    #         st.pyplot(fig1)

    #     with col2 :
    #         st.subheader("Comparatif des métriques par classe")

    #         tuples = [tuple(c.split("-",1)) for c in df_classes_synt.columns[1:]]

    #         # 2) Création du MultiIndex
    #         multi_index = pd.MultiIndex.from_tuples(tuples, names=["Modèle", "Métrique"])

    #         # 3) On reconstruit un DataFrame avec ce MultiIndex
    #         df_plot = pd.DataFrame(df_classes_synt.iloc[:, 1:].to_numpy(),
    #                             index=df_classes_synt["Classe"],
    #                             columns=multi_index)

    #         # --- Heatmap ---
    #         fig2, ax2 = plt.subplots(figsize=(7, 4))
    #         sns.heatmap(df_plot, annot=True, fmt=".1f", cmap="YlGnBu",
    #                     cbar=False, linewidths=0.5, linecolor="white", ax=ax2)

    #         # Masquer l'affichage des colonnes en bas
    #         ax2.set_xticks([])
    #         ax2.set_xticklabels([])
    #         for label in ax2.get_xticklabels():
    #             label.set_visible(False)

    #         # Réduire la taille des labels des classes
    #         ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=10)

    #         # Ajouter les ticks de métriques manuellement (en haut)
    #         metrics = df_plot.columns.get_level_values(1)
    #         ax2.set_xticks(np.arange(len(metrics)) + 0.5)
    #         ax2.set_xticklabels(metrics, rotation=0, ha="center", fontsize=10)
    #         ax2.xaxis.set_ticks_position("top")
    #         ax2.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    #         # Ajouter les labels de modèles au-dessus des métriques
    #         level0 = df_plot.columns.get_level_values(0)
    #         unique_models = level0.unique()
    #         for model in unique_models:
    #             idxs = np.where(level0 == model)[0]
    #             x_center = idxs.mean() + 0.5
    #             ax2.text(x_center, -0.7, model,
    #                     ha='center', va='bottom', fontsize=10, fontweight='bold', transform=ax2.transData)

    #         ax2.set_ylabel("Classe")
    #         st.pyplot(fig2)

    #     st.markdown("""
    #     """) 



if __name__ == "__main__":
    main()




