# K-Means Aprendizaje Semi-Supervisado

## Descripción General

Este notebook implementa técnicas de **aprendizaje semi-supervisado** usando el algoritmo **K-Means** en el dataset **bloodMNIST**. El objetivo es entrenar modelos de clasificación efectivos utilizando solo una pequeña fracción de datos etiquetados manualmente.

## Contenido del Notebook

### 1. **Carga y Preparación de Datos**
- Carga del dataset bloodMNIST (imágenes de células de sangre)
- Separación en conjuntos de entrenamiento, validación y prueba
- Redimensionamiento de imágenes a formato plano (flatten) para procesamiento con K-Means
- Consolidación de datos de entrenamiento y validación

**Variables clave:**
- `X_train`: 3200 imágenes de entrenamiento (flattened)
- `X_val`: 800 imágenes de validación (flattened)
- `X_test`: 1024 imágenes de prueba (flattened)
- `y_train`, `y_val`, `y_test`: etiquetas correspondientes

---

### 2. **Clustering con K-Means (k=50)**
- Aplicación del algoritmo K-Means con 50 clusters
- Transformación de los datos: cálculo de distancias de cada muestra a cada centroide
- Matriz de distancias de forma (3200, 50)

```
kmeans = KMeans(n_clusters=50, random_state=42)
X_digits_dist = kmeans.fit_transform(X_train)
```

**Resultado:** 50 grupos automáticos sin usar etiquetas

---

### 3. **Identificación de Imágenes Representativas**
- Extracción de la imagen más cercana a cada centroide (representante de cluster)
- Selección de 50 imágenes más representativas del dataset
- Visualización de estas imágenes en una cuadrícula de 5×10

**Proceso:**
1. Encontrar índice del punto más cercano a cada centroide
2. Extraer esas imágenes del conjunto de entrenamiento
3. Mostrar visualmente las 50 imágenes seleccionadas

---

### 4. **Etiquetado Manual de Representantes**
- Anotación manual de las 50 imágenes representativas (en el notebook se realiza "trampa" usando las etiquetas reales disponibles)
- Creación de un pequeño dataset etiquetado con solo 50 muestras

```python
y_representative_digits = y_train[idxs]
```

---

### 5. **Entrenamiento de Clasificador con Datos Limitados**
- Entrenamiento de Regresión Logística Multiclase usando solo 50 imágenes representativas
- Evaluación en el conjunto de prueba

**Resultados:**
- Precisión conseguida: ~44% en el conjunto de prueba
- Comparación: solo 50 imágenes anotadas vs. imágenes aleatorias

**Comparativa con selección aleatoria:**
- 50 imágenes representativas: mejor desempeño
- 50 imágenes aleatorias: peor desempeño
- **Conclusión:** La calidad de los datos etiquetados es más importante que la cantidad

---

### 6. **Propagación Automática de Etiquetas**
- Asignación de la etiqueta de cada representante a todas las imágenes de su cluster
- Creación de un dataset de 3200 imágenes con etiquetas propagadas (aunque con posible ruido)

```python
y_train_propagated = np.empty(len(X_train))
for i in range(k):
    y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]
```

**Ventaja:** Etiquetado automático de miles de imágenes
**Desventaja:** Introducción de ruido si algunos clusters tienen baja pureza

---

### 7. **Entrenamiento Iterativo con Propagación**
- Entrenamiento de nuevo clasificador usando 1000 imágenes con etiquetas propagadas
- Evaluación en el conjunto de prueba
- Análisis de cómo el ruido afecta al modelo

---

### 8. **Aprendizaje Activo (Active Learning)**
- Identificación de muestras con menor confianza (predicciones débiles)
- Selección de las 50 imágenes donde el modelo tiene más dudas
- Visualización de estas imágenes problemáticas

**Proceso:**
1. Obtener probabilidades de predicción para cada clase
2. Encontrar la máxima probabilidad por predicción
3. Ordenar por confianza (menor confianza primero)
4. Seleccionar las 50 más inciertas

```python
probas = log_reg3.predict_proba(X_train[:1000])
labels_ixs = np.argmax(probas, axis=1)
labels = np.array([proba[ix] for proba, ix in zip(probas, labels_ixs)])
sorted_ixs = np.argsort(labels)  # Ordenar por confianza
X_lowest = X_train[:1000][sorted_ixs[:k]]
```

---

### 9. **Corrección Manual y Mejora Iterativa**
- Etiquetado correcto manual de las 50 imágenes con menor confianza
- Re-entrenamiento del modelo con datos corregidos
- Mejora del desempeño del clasificador

```python
y_train2 = y_train_propagated[:1000].copy()
y_train2[sorted_ixs[:k]] = y_lowest  # Corregir etiquetas incorrectas
log_reg5.fit(X_train[:1000], y_train2)
```

**Resultado:** Mejora en la precisión del modelo

---

## Flujo del Aprendizaje Semi-Supervisado

```
1. Dataset sin etiquetas → K-Means Clustering (50 clusters)
                                    ↓
2. Identificar representantes → Anotar manualmente 50 imágenes
                                    ↓
3. Entrenar clasificador con 50 muestras → Precisión ~44%
                                    ↓
4. Propagar etiquetas → Dataset de 3200 muestras etiquetadas (con ruido)
                                    ↓
5. Entrenar con datos propagados → Evaluación
                                    ↓
6. Aprendizaje Activo → Identificar ejemplos más inciertos
                                    ↓
7. Corregir manualmente → Re-entrenar modelo
                                    ↓
8. Mejora iterativa del clasificador
```

---

## Técnicas Utilizadas

### **K-Means Clustering**
- Agrupamiento no supervisado de 3200 imágenes en 50 clusters
- Identificación de patrones sin etiquetas previas

### **Semi-Supervised Learning**
- Combinación de pocos datos etiquetados con muchos datos sin etiquetar
- Propagación de etiquetas desde representantes a clusters completos

### **Active Learning**
- Selección inteligente de muestras para anotar manualmente
- Enfoque en ejemplos donde el modelo tiene mayor incertidumbre
- Mejora iterativa del clasificador con mínimo esfuerzo de anotación

### **Logistic Regression Multiclase**
- Clasificador entrenado con datos semi-supervisados
- Soporte para múltiples clases de células sanguíneas

---

## Resultados Clave

| Estrategia | Datos Etiquetados | Precisión en Test |
|-----------|------------------|------------------|
| 50 imágenes representativas | 50 | ~44% |
| 50 imágenes aleatorias | 50 | Menor que arriba |
| Propagación automática (1000) | ~1000 | Moderada |
| Aprendizaje activo iterativo | 1000 + correcciones | Mejorada |

---

## Conclusiones

1. **Calidad > Cantidad:** Seleccionar 50 imágenes representativas es mejor que 50 aleatorias
2. **Propagación automática:** Etiquetado rápido de grandes volúmenes, pero introduce ruido
3. **Aprendizaje activo es efectivo:** Corregir ejemplos donde el modelo duda mejora el desempeño significativamente
4. **Aplicabilidad práctica:** Ideal cuando:
   - Tenemos muchos datos sin etiquetar
   - Etiquetar manualmente es costoso
   - Podemos identificar patrones automáticos iniciales

---

## Dependencias

```
numpy
matplotlib
scikit-learn
```

---

## Dataset: bloodMNIST

- **Descripción:** Imágenes de células de sangre (28×28×3 píxeles)
- **Clases:** Múltiples tipos de células sanguíneas
- **Archivo:** `bloodmnist.npz`
- **División:** Train, Validation, Test

---

## Aplicaciones Prácticas

Este enfoque es útil en escenarios como:
- Clasificación de imágenes médicas (donde el etiquetado requiere expertos)
- Detección de anomalías
- Segmentación de clientes con datos parcialmente etiquetados
- Análisis de textos con pocas etiquetas iniciales
- Cualquier tarea donde etiquetar manualmente es costoso y tedioso
