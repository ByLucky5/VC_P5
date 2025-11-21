import os
import shutil
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

# -------------------------
# PATHS
# -------------------------
SOURCE = "./dataset"
DEST = "./data"

# Crear carpetas necesarias
folders = [
    f"{DEST}/images",
    f"{DEST}/emotions",
    f"{DEST}/Prueba"
]

for f in folders:
    os.makedirs(f, exist_ok=True)

# -------------------------
# Mapeo de etiquetas
# -------------------------
labels = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "pain": 4,
    "sad": 5
}

# Crear subcarpetas en images
for lbl in labels.keys():
    os.makedirs(f"{DEST}/images/{lbl}", exist_ok=True)

# -------------------------
# Procesar cada clase
# -------------------------
for folder_name, label in labels.items():
    print(f"Procesando clase: {folder_name}")
    
    # Lista todas las imágenes de la carpeta
    images = glob(f"{SOURCE}/{folder_name}/*")
    
    # Dividir train/test 80/20
    train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
    
    # -------------------
    # Guardar TRAIN
    # -------------------
    train_rows = []
    for img_path in train_imgs:
        fname = os.path.basename(img_path)
        dest_path = f"{DEST}/images/{folder_name}/{fname}"
        shutil.copy(img_path, dest_path)
        train_rows.append([f"{folder_name}/{fname}", label])
    
    # CSV por clase
    pd.DataFrame(train_rows, columns=["filename", "label"]).to_csv(
        f"{DEST}/emotions/{folder_name}.csv.txt",
        index=False
    )
    
    # -------------------
    # Guardar TEST
    # -------------------
    test_rows = []
    for img_path in test_imgs:
        fname = os.path.basename(img_path)
        dest_path = f"{DEST}/images/{folder_name}/{fname}"
        shutil.copy(img_path, dest_path)
        test_rows.append([f"{folder_name}/{fname}", label])
    
    pd.DataFrame(test_rows, columns=["filename", "label"]).to_csv(
        f"{DEST}/Prueba/test_{folder_name}.csv.txt",
        index=False
    )

print("¡Dataset procesado correctamente!")
