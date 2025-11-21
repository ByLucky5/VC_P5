# Práctica 5 - Detector de Emociones con Filtros en Tiempo Real

Práctica realizada por Lucía Motas Guedes y Raúl Marrero Marichal.

Este proyecto implementa un **sistema de detección de emociones faciales en tiempo real** mediante **transfer learning con ResNet18** y la aplicación de **filtros PNG con transparencia**, capaces de:

* Detectar 4 emociones: `pain`, `disgust`, `fear` y `happy`.
* Aplicar filtros específicos a cada emoción sobre la cara detectada.
  * `pain`: cuernos rojos en la parte superior del rostro.
  * `disgust`: corazones en las mejillas.
  * `fear`: se utilizó un png de un ojo irritado y se utilizó para ambos ojos.
  * `happy`: nariz y bigotes de gato en la mitad del rostro.
* Mostrar la emoción detectada sobre la cara en tiempo real.
* Evaluar el modelo con métricas y mostrar la **matriz de confusión** en porcentaje.

---

## Estructura del Proyecto

```bash
├── data/
│   ├── filters/                  # PNGs de filtros con fondo transparente
│       ├── pain.png
│       ├── disgust.png
│       ├── fear.webp
│       └── happy.png
│   ├── images/               # Todas las imágenes originales
│   ├── emotions/             # CSV con imágenes de entrenamiento por emoción
│   └── Prueba/               # CSV con imágenes de test por emoción
│
├── modelo_emociones.pth      # Modelo entrenado con ResNet18
├── VC_P5.ipynb                  # Cuaderno con el desarrollo de la práctica
├── divide.py                 # Script para organizar los datos de kaggle
└── README.md
```

> Nota: Se utiliza **OpenCV** para la captura de cámara y detección de caras (Haar Cascade).

---

## 1. Configuración del Entorno

Requiere Python 3.9+ y, opcionalmente, soporte CUDA para GPU.

```bash
# === Crear el entorno virtual ===
python -m venv VC_P5

# === Activar el entorno ===
.\VC_P5\Scripts\activate

# === Instalar PyTorch compatible con CUDA 12.x === pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 # o cu126

# === Instalar dependencias ===
pip install opencv-python matplotlib pandas scikit-learn pillow numpy
```

---

## 2. Dataset

Se utilizó el dataset de Kaggle: [Sentiment Images Classifier](https://www.kaggle.com/datasets/yousefmohamed20/sentiment-imagesclassifier).

### Organización del dataset

```bash
dataset/
├── images/               # imágenes originales por emoción
├── emotions/             # CSVs con rutas y etiquetas para entrenamiento
└── Prueba/               # CSVs con rutas y etiquetas para test
```

> Los CSVs contienen dos columnas: `filename` (ruta de la imagen) y `label` (entidad numérica de la emoción).
> Se filtraron solo 4 emociones para este proyecto: `pain`, `disgust`, `fear` y `happy`.
> Por tanto, se mapean para solo utilizar las 4 elegidas.

---

## 3. Entrenamiento del Modelo

* Se utilizó **ResNet18 preentrenado** mediante **transfer learning**.
* Se congelaron todas las capas excepto las últimas 3 y la capa fully connected.
* La última capa se reemplazó por una capa lineal adaptada a 4 clases (emociones).
* Optimización: `Adam`, función de pérdida `CrossEntropyLoss`.
* Entrenamiento: 5 epochs, batch size 64.

```python
# Ejemplo de modificación de la última capa
model = models.resnet18(pretrained=True)
for layer in list(model.children())[:-3]:
    for param in layer.parameters():
        param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 4)
```

---

## 4. Evaluación del Modelo

* Cálculo de **accuracy** sobre el conjunto de test.
* Generación de la **matriz de confusión en porcentaje**:

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(all_labels, all_preds)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=emociones)
disp.plot(cmap='Blues', values_format=".2f")
```

---

## 5. Detección de Emociones en Tiempo Real

* **Webcam** + **Haar Cascade** para detectar caras.
* **Transformación de imágenes**: resize a 128x128, normalización.
* **Predicción de emoción** mediante el modelo entrenado.
* **Aplicación de filtros** PNG con transparencia:

| Emoción   | Filtro aplicado      | Región en la cara  |
| --------- | -------------------- | ------------------ |
| `pain`    | cuernos rojos        | superior           |
| `disgust` | corazones            | mejillas           |
| `fear`    | ojos irritados       | ambos ojos         |
| `happy`   | nariz y bigotes gato | zona media (nariz) |

* Los filtros con alpha se superponen respetando la transparencia.
* En `fear`, el filtro se duplica para ambos ojos.

---

## 6. Consideraciones

* Se mantiene la **detección de caras con Haar Cascade** porque Mediapipe puede fallar en ciertos ángulos de webcam.
* Los filtros están ajustados proporcionalmente a la cara detectada y posicionados para lucir natural.
* Los PNGs se cargan con canal alpha para mantener transparencia.
