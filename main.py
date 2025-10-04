# Création d'une version corrigée et nettoyée du notebook "Livrable2.ipynb"
import nbformat, os, math, json
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

fixed_nb = new_notebook()
fixed_nb['cells'] = []

# 1. Title/intro
fixed_nb['cells'].append(new_markdown_cell("# Livrable 2 — Traitement d'images\n\nNotebook nettoyé et fonctionnel pour le débruitage d'images avec un autoencodeur convolutif.\n\n**Remarques**: ce notebook utilise un pipeline tf.data pour être mémoire-efficient. Ajuste `data_dir`, `MAX_IMAGES` et `BATCH_SIZE` selon ta machine."))

# 2. Imports
fixed_nb['cells'].append(new_code_cell("""# Imports essentiels
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from perlin_noise import PerlinNoise  # pip install perlin-noise
from skimage.metrics import peak_signal_noise_ratio as psnr

print('TensorFlow version:', tf.__version__)"""))

# 3. Params and dataset loading (memory efficient)
fixed_nb['cells'].append(new_markdown_cell("## 1) Paramètres et chargement du dataset\n\nChange `data_dir` pour le chemin de ton dossier d'images. Si toutes les images sont dans un seul dossier (pas de sous-classes), on utilise `labels=None` et on gère le split manuellement."))

fixed_nb['cells'].append(new_code_cell("""# Paramètres - adapte si nécessaire
data_dir = r'C:/Users/REAL/PycharmProjects/Bloc_IA_FISE5/Datasett/train2014'  # <-- adapte ce chemin
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

# Option pour limiter le dataset pour prototypage (None = utiliser tout)
MAX_IMAGES = 20000  # mettre None pour tout utiliser

# Vérification du chemin
if not os.path.exists(data_dir):
    raise FileNotFoundError(f\"Le dossier data_dir n'existe pas: {data_dir}\")"""))

# 4. Create dataset and optional limit
fixed_nb['cells'].append(new_code_cell("""# Chargement sans labels (images seules)
raw_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels=None,
    label_mode=None,
    shuffle=True,
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Convertir nombre de batches en nombre d'images et limiter si demandé
if MAX_IMAGES is not None:
    # nombre de batches nécessaires
    batches_to_keep = math.ceil(MAX_IMAGES / BATCH_SIZE)
    raw_ds = raw_ds.take(batches_to_keep)
    print(f\"Limitation activée : ~{batches_to_keep * BATCH_SIZE} images (≈ {MAX_IMAGES} demandé)\")
else:
    print(\"Utilisation de tout le dataset (aucune limitation)\")"""))

# 5. Split train/test
fixed_nb['cells'].append(new_markdown_cell("## 2) Split train / validation\n\nOn split le dataset déjà chargé (qui est mélangé). Ici 80% train, 20% validation."))

fixed_nb['cells'].append(new_code_cell("""# Estimer le nombre de batches et split
total_batches = tf.data.experimental.cardinality(raw_ds).numpy()
if total_batches == tf.data.experimental.INFINITE_CARDINALITY:
    raise RuntimeError('cardinality infinite — vérifie le dataset')

train_batches = int(0.8 * total_batches)
val_batches = total_batches - train_batches

train_ds = raw_ds.take(train_batches)
val_ds = raw_ds.skip(train_batches)

print(f\"Batches totaux: {total_batches}, train: {train_batches}, val: {val_batches}\")"""))

# 6. Preprocessing and noise addition (efficient)
fixed_nb['cells'].append(new_markdown_cell("## 3) Normalisation et ajout de bruit (pipeline tf.data)\n\nOn normalise dans [0,1] puis on crée deux pipelines : clean et noisy. Le bruit est ajouté de façon vectorisée (pas de conversion complète en NumPy)."))

fixed_nb['cells'].append(new_code_cell("""# paramètres du bruit
NOISE_FACTOR = 0.3

def add_gaussian_noise(images):
    noise = tf.random.normal(shape=tf.shape(images), mean=0.0, stddev=1.0)
    noisy = images + NOISE_FACTOR * noise
    noisy = tf.clip_by_value(noisy, 0.0, 1.0)
    return noisy

# Normalisation
def normalize(images):
    images = tf.cast(images, tf.float32) / 255.0
    return images

# Preprocess: normalize, cache small, prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_clean = train_ds.map(lambda x: normalize(x), num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
train_noisy = train_ds.map(lambda x: normalize(x), num_parallel_calls=AUTOTUNE).map(lambda x: add_gaussian_noise(x), num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

val_clean = val_ds.map(lambda x: normalize(x), num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
val_noisy = val_ds.map(lambda x: normalize(x), num_parallel_calls=AUTOTUNE).map(lambda x: add_gaussian_noise(x), num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

# Dataset_livrable2 prêt pour l'entraînement (zip noisy->clean)
train_dataset = tf.data.Dataset_livrable2.zip((train_noisy, train_clean))
val_dataset = tf.data.Dataset_livrable2.zip((val_noisy, val_clean))

# Calcul du nombre d'images approximatif
approx_train_images = train_batches * BATCH_SIZE
approx_val_images = val_batches * BATCH_SIZE
print(f\"Approx images — train: {approx_train_images}, val: {approx_val_images}\")"""))

# 7. Visualize some noisy vs clean images
fixed_nb['cells'].append(new_markdown_cell("## 4) Visualisation : quelques images propres vs bruitées"))

fixed_nb['cells'].append(new_code_cell("""import matplotlib.pyplot as plt

def show_examples(clean_ds, noisy_ds, n=5):
    # Récupère un batch (premier) de chaque dataset
    clean_batch = next(iter(clean_ds))
    noisy_batch = next(iter(noisy_ds))
    plt.figure(figsize=(12, 5))
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(clean_batch[i].numpy())
        plt.title('Propre')
        plt.axis('off')

        ax = plt.subplot(2, n, i+1+n)
        plt.imshow(noisy_batch[i].numpy())
        plt.title('Bruitée')
        plt.axis('off')
    plt.show()

show_examples(val_clean, val_noisy, n=5)"""))

# 8. Model: Convolutional Autoencoder (simple)
fixed_nb['cells'].append(new_markdown_cell("## 5) Modèle : Autoencodeur convolutif\n\nModèle simple, à améliorer si besoin."))

fixed_nb['cells'].append(new_code_cell("""input_shape = IMG_SIZE + (3,)
inputs = layers.Input(shape=input_shape)

# Encodeur
x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
x = layers.MaxPooling2D(2, padding='same')(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D(2, padding='same')(x)

# Décodeur
x = layers.Conv2D(64, 3, activation='relu', padding='same')(encoded)
x = layers.UpSampling2D(2)(x)
x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
x = layers.UpSampling2D(2)(x)
decoded = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)

autoencoder = models.Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.summary()"""))

# 9. Training cell with steps per epoch safe calculation
fixed_nb['cells'].append(new_markdown_cell("## 6) Entraînement\n\nOn calcule `steps_per_epoch` pour éviter d'attendre l'épuisement complet des datasets si nécessaire."))

fixed_nb['cells'].append(new_code_cell("""EPOCHS = 20

# Calculer steps par epoch
train_steps = tf.data.experimental.cardinality(train_dataset).numpy()
val_steps = tf.data.experimental.cardinality(val_dataset).numpy()
print('Train steps (batches):', train_steps, 'Val steps (batches):', val_steps)

history = autoencoder.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_data=val_dataset,
    validation_steps=val_steps
)"""))

# 10. Evaluation: afficher résultats et PSNR
fixed_nb['cells'].append(new_markdown_cell("## 7) Évaluation : affichage et métrique PSNR"))

fixed_nb['cells'].append(new_code_cell("""# Prédictions sur un batch de validation
noisy_batch, clean_batch = next(iter(val_dataset))
preds = autoencoder.predict(noisy_batch)

# Affichage
n = min(5, noisy_batch.shape[0])
plt.figure(figsize=(12,6))
for i in range(n):
    ax = plt.subplot(3, n, i+1)
    plt.imshow(noisy_batch[i].numpy())
    plt.title('Bruitée')
    plt.axis('off')

    ax = plt.subplot(3, n, i+1+n)
    plt.imshow(preds[i])
    plt.title('Débruitée')
    plt.axis('off')

    ax = plt.subplot(3, n, i+1+2*n)
    plt.imshow(clean_batch[i].numpy())
    plt.title('Origine')
    plt.axis('off')
plt.show()

# Calcul PSNR moyen (sur ce batch)
psnr_values = [psnr(clean_batch[i].numpy(), preds[i]) for i in range(n)]
print('PSNR (moyenne sur batch):', sum(psnr_values)/len(psnr_values))"""))

# 11. Save notebook to file
out_path = "Projet/Livrable_2/Livrable2_fixed.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbformat.write(fixed_nb, f)