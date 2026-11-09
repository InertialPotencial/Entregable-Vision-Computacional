# TECNOFORMS - RECONOCIMIENTO DE DIGITOS MNIST

import os
import joblib
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score
)

# CONFIGURACION

np.random.seed(42)         # Establecemos una semilla(seed) para mantener resultados reproducibles
tf.random.set_seed(42)     # y facilitar la comparacion entre los modelos en cada ejecucion.

print("\nTECNOFORMS - RECONOCIMIENTO MNIST")

# 1. CARGA Y PREPROCESAMIENTO

print("\nCargando dataset MNIST...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() #Cargamos los datos en las variable de prueba y entrenamiento

# Normalizamos los pixeles para facilitar el aprendizaje.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Adaptamos las imagenes al formato de entrada de CNN.
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Reservamos datos para comprobar el aprendizaje.
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,test_size=0.2,random_state=42,stratify=y_train)

print("Entrenamiento:", x_train.shape)
print("Validacion:", x_val.shape)
print("Prueba:", x_test.shape)


# 2. VISUALIZACION DEL DATASET CON MATPLOTLIB

plt.figure(figsize=(9, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i].reshape(28, 28), cmap="gray")
    plt.title(f"Digito: {y_train[i]}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("resultados/digitos_mnist.png", dpi=300)
plt.close()


# 3. FUNCION GENERAL DE EVALUACION

def evaluar_modelo(y_real, y_pred, nombre):

    accuracy = accuracy_score(y_real, y_pred)
    reporte = classification_report(y_real, y_pred,digits=4,zero_division=0)

    print(f"\nRESULTADOS {nombre}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(reporte)

    # La matriz permite identificar los digitos confundidos.
    cm = confusion_matrix(y_real, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matriz de confusion - {nombre}")
    plt.xlabel("Prediccion")
    plt.ylabel("Valor real")
    plt.tight_layout()

    plt.savefig(
        f"resultados/matriz_confusion_{nombre.lower()}.png",
        dpi=300
    )
    plt.close()

    with open(
        f"resultados/reporte_{nombre.lower()}.txt",
        "w",
        encoding="utf-8"
    ) as archivo:
        archivo.write(reporte)

    return accuracy


# 4. MODELO CNN

print("\nEntrenando CNN...")

# Conservamos la arquitectura original de tres capas convolucionales.
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss="sparse_categorical_crossentropy",metrics=["accuracy"])

# Evitamos entrenar innecesariamente y guardamos la mejor version.
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=4,restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=2,min_lr=0.00001),
    tf.keras.callbacks.ModelCheckpoint(     #Guardamos los resultado en una carpeta apartada
        "modelos/mejor_modelo_mnist.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

history = model.fit(x_train, y_train,epochs=20,batch_size=64,validation_data=(x_val, y_val),callbacks=callbacks,verbose=1)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
model.save("modelos/modelo_mnist.keras")
pred_cnn = np.argmax(model.predict(x_test, verbose=0), axis=1)
accuracy_cnn = evaluar_modelo(y_test, pred_cnn, "CNN")


# 5. GRAFICAS DE ENTRENAMIENTO

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Entrenamiento")
plt.plot(history.history["val_accuracy"], label="Validacion")
plt.title("Exactitud")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Entrenamiento")
plt.plot(history.history["val_loss"], label="Validacion")
plt.title("Perdida")
plt.legend()

plt.tight_layout()
plt.savefig("resultados/graficas_entrenamiento.png", dpi=300)
plt.close()


# 6. MODELO KNN

print("\nEntrenando KNN...")

# Convertimos cada imagen en una fila de 784 valores.
x_train_knn = x_train.reshape(-1, 784)
x_val_knn = x_val.reshape(-1, 784)

# Utilizamos muestras para reducir el tiempo de procesamiento.
x_train_knn, _, y_train_knn, _ = train_test_split(x_train_knn, y_train,train_size=0.2,random_state=42,stratify=y_train)
x_val_knn, _, y_val_knn, _ = train_test_split(x_val_knn, y_val,train_size=0.2,random_state=42,stratify=y_val)

# Entrenamos KNN con 5 vecinos y distancia euclidiana.
knn = KNeighborsClassifier(n_neighbors=5,weights="distance",p=2)
knn.fit(x_train_knn, y_train_knn)

# Realizamos las predicciones.
pred_knn = knn.predict(x_val_knn)

# Evaluamos los resultados.
accuracy_knn = evaluar_modelo(y_val_knn, pred_knn, "KNN")

# Guardamos el modelo entrenado.
joblib.dump(knn, "modelos/modelo_knn.pkl")

# 7. COMPARACION CNN Y KNN

# Evaluamos ambos modelos sobre las mismas imagenes.
x_val_cnn = x_val_knn.reshape(-1, 28, 28, 1)

pred_cnn_val = np.argmax(model.predict(x_val_cnn, verbose=0),axis=1)
accuracy_cnn_val = accuracy_score(y_val_knn, pred_cnn_val)

print("\nCOMPARACION DE MODELOS")
print(f"CNN: {accuracy_cnn_val * 100:.2f}%")
print(f"KNN: {accuracy_knn * 100:.2f}%")

mejor_modelo = ("CNN" if accuracy_cnn_val >= accuracy_knn else "KNN")


# 8. MODELO K-MEANS

print("\nEntrenando K-Means...")

# Seleccionamos una muestra de imagenes para trabajar mas rapido.
x_kmeans, _, y_kmeans, _ = train_test_split(x_train.reshape(-1, 784),y_train,train_size=0.2,random_state=42,stratify=y_train)

# Reducimos las 784 caracteristicas a 50 para facilitar el agrupamiento.
pca = PCA(n_components=50, random_state=42)
x_pca = pca.fit_transform(x_kmeans)

# Buscamos organizar las imagenes en diez grupos.
kmeans = KMeans(n_clusters=10,random_state=42,n_init=10,max_iter=300)

clusters = kmeans.fit_predict(x_pca)


# 9. EVALUACION K-MEANS

ari = adjusted_rand_score(y_kmeans, clusters)
nmi = normalized_mutual_info_score(y_kmeans, clusters)

silhouette = silhouette_score(
    x_pca,
    clusters,
    sample_size=2000,
    random_state=42
)

print("\nRESULTADOS K-MEANS")
print(f"ARI: {ari:.4f}")
print(f"NMI: {nmi:.4f}")
print(f"Silhouette: {silhouette:.4f}")


# 10. GRAFICA COMPARATIVA

plt.figure(figsize=(7, 5))

barras = plt.bar(["CNN", "KNN"],[accuracy_cnn_val * 100, accuracy_knn * 100],color=["royalblue", "orange"])

plt.title("Comparacion de algoritmos - TecnoForms")
plt.ylabel("Exactitud (%)")
plt.ylim(0, 105)
plt.grid(axis="y", alpha=0.3)

for barra in barras:
    plt.text(
        barra.get_x() + barra.get_width() / 2,
        barra.get_height() + 1,
        f"{barra.get_height():.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig("resultados/comparacion_modelos.png", dpi=300)
plt.close()


# 11. GUARDADO DE RESULTADOS

with open(
    "resultados/comparacion_algoritmos.txt",
    "w",
    encoding="utf-8"
) as archivo:

    archivo.write("RESULTADOS TECNOFORMS\n\n")
    archivo.write(f"CNN prueba: {test_accuracy:.4f}\n")
    archivo.write(f"CNN validacion: {accuracy_cnn_val:.4f}\n")
    archivo.write(f"KNN validacion: {accuracy_knn:.4f}\n")
    archivo.write(f"Mejor clasificador: {mejor_modelo}\n\n")
    archivo.write(f"K-MEANS ARI: {ari:.4f}\n")
    archivo.write(f"K-MEANS NMI: {nmi:.4f}\n")
    archivo.write(f"K-MEANS Silhouette: {silhouette:.4f}\n")


with open(
    "resultados/resultados_kmeans.txt",
    "w",
    encoding="utf-8"
) as archivo:

    archivo.write("RESULTADOS K-MEANS\n")
    archivo.write(f"Imagenes: {len(x_kmeans)}\n")
    archivo.write(f"Grupos: {kmeans.n_clusters}\n")
    archivo.write(f"ARI: {ari:.4f}\n")
    archivo.write(f"NMI: {nmi:.4f}\n")
    archivo.write(f"Silhouette: {silhouette:.4f}\n")


# 12. RESUMEN FINAL

print("\n===== TECNOFORMS FINALIZADO =====")
print(f"CNN (prueba): {test_accuracy * 100:.2f}%")
print(f"CNN (validacion): {accuracy_cnn_val * 100:.2f}%")
print(f"KNN (validacion): {accuracy_knn * 100:.2f}%")
print(f"K-MEANS ARI: {ari:.4f}")
print(f"Mejor clasificador: {mejor_modelo}")
print("\nModelos y resultados guardados correctamente.")