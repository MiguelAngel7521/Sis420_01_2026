# Regresión Logística — FIFA 17 + FIFA 18

Clasificación binaria para predecir si un jugador de FIFA es **élite** (Overall ≥ 80) usando Regresión Logística implementada desde cero con NumPy.

---

## Descripción del Proyecto

El modelo analiza las estadísticas de **35,487 jugadores** de FIFA 17 y FIFA 18 para predecir si un jugador alcanza el nivel élite. Se implementan y comparan dos métodos de optimización, con una división estricta de datos para validación real del modelo.

### Variable objetivo
| Valor | Significado | Condición |
|-------|-------------|-----------|
| `1` | Jugador Élite | >0.5 |
| `0` | Jugador No Élite | <0.5 |

---

## Dataset

- **Fuente:** FIFA 17 + FIFA 18 combinados
- **Muestras:** m = 35,487 jugadores
- **Features:** n = 40 variables numéricas

| # | Feature | Descripción |
|---|---------|-------------|
| 1 | `Age` | Edad del jugador |
| 2 | `Potential` | Potencial máximo estimado |
| 3–11 | `Crossing`, `Finishing`, `HeadingAccuracy`, `ShortPassing`, `Volleys`, `Dribbling`, `Curve`, `FKAccuracy`, `LongPassing` | Habilidades técnicas |
| 12–20 | `BallControl`, `Acceleration`, `SprintSpeed`, `Agility`, `Reactions`, `Balance`, `ShotPower`, `Jumping`, `Stamina` | Físico y control |
| 21–31 | `Strength`, `LongShots`, `Aggression`, `Interceptions`, `Positioning`, `Vision`, `Penalties`, `Composure`, `Marking`, `StandingTackle`, `SlidingTackle` | Defensa y mentalidad |
| 32–36 | `GKDiving`, `GKHandling`, `GKKicking`, `GKPositioning`, `GKReflexes` | Atributos de portero |
| 37–40 | `International Reputation`, `Weak Foot`, `Skill Moves`, `FIFA_Year` | Reputación y año |

---

## División Train / Test — 80% / 20%

```
Total: 35,487 jugadores
├── Train (80%): ~28,389 jugadores  ← usado para entrenar
└── Test  (20%):  ~7,097 jugadores  ← solo para evaluar (nunca visto en entrenamiento)
```

### Reglas de aislamiento aplicadas
- Los índices se mezclan aleatoriamente con `np.random.seed(42)` para reproducibilidad
- `μ` y `σ` de normalización se calculan **solo sobre train** y se aplican al test
- Los datos de test **no participan** en ninguna iteración del entrenamiento ni en la selección de hiperparámetros

---

## Modelos Implementados

### Modelo 1 — Descenso por el Gradiente

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}\left[ y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)})) \right]$$

$$\theta := \theta - \frac{\alpha}{m} X^T(h_\theta(X) - y)$$

| Hiperparámetro | Valor |
|----------------|-------|
| Learning rate α | 0.1 |
| Iteraciones | 300 |
| θ inicial | Ceros |

### Modelo 2 — Optimización TNC (scipy)

Método de Newton Truncado que calcula costo y gradiente analítico simultáneamente. Converge más rápido y preciso sin necesidad de ajustar α manualmente.

| Parámetro | Valor |
|-----------|-------|
| Método | TNC (Truncated Newton Conjugate) |
| Max iteraciones | 1000 |
| Gradiente | Analítico (jac=True) |

---

## Evaluación del Modelo

### Métricas calculadas sobre el Test Set (20%)

| Métrica | Descripción |
|---------|-------------|
| **Precisión (Accuracy)** | % de predicciones correctas |
| **Precisión por clase** | Precision, Recall, F1-Score |
| **AUC-ROC** | Área bajo la curva ROC (1.0 = perfecto) |
| **Matriz de Confusión** | VP, VN, FP, FN detallados |

### Tipos de error en la matriz de confusión

| | Predicho: No Élite | Predicho: Élite |
|--|-------------------|-----------------|
| **Real: No Élite** | VN | FP |
| **Real: Élite** | FN | VP |

---

## Uso en Google Colab

### Requisitos
```
numpy  pandas  matplotlib  scipy  sklearn  seaborn
```

### Pasos
1. Subir `FIFA_17_18_combined.csv` a Google Drive
2. Abrir `FIFA_Regresion_Logistica_v2.ipynb` en Colab
3. Ajustar la ruta en la celda 2:
```python
RUTA = '/content/drive/MyDrive/ruta/FIFA_17_18_combined.csv'
```
4. Ejecutar todo: `Ctrl + F9`

### Predecir un jugador nuevo
Modificar el diccionario en la celda 14:
```python
jugador = {
    'Age': 25, 'Potential': 85, 'Reactions': 82,
    # ... resto de las 40 features
}
predecir_jugador(jugador, features, X_mean, X_std, theta_opt)
# Output: "Probabilidad de ser élite: 78.3%  ->  ELITE"
```

---

## Estructura del Cuadernillo

| Celda | Contenido |
|-------|-----------|
| 1 | Librerías y montaje de Google Drive |
| 2 | Carga del CSV, selección de features, creación del target binario |
| 3 | **División Train/Test 80/20** sin solapamiento (`seed=42`) |
| 4 | Normalización z-score (μ/σ calculados solo en train) |
| 5 | Visualización Reactions vs Potential (train y test) |
| 6 | Función sigmoide con gráfica |
| 7 | Función de costo Binary Cross-Entropy |
| 8 | Modelo 1 — Gradiente Descendente + curvas de costo y precisión |
| 9 | Modelo 2 — Optimización TNC (scipy) |
| 10 | **Evaluación Train vs Test** (detección de overfitting) |
| 11 | Matrices de Confusión sobre Test Set |
| 12 | Curva ROC y AUC sobre Test Set |
| 13 | Features más importantes (coeficientes θ) |
| 14 | Predicción de jugadores nuevos |
| 15 | Resumen final comparativo |

---

## Validación Correcta — Sin Data Leakage

El punto más importante del proyecto es garantizar que el modelo se evalúe de forma honesta:

```python
# CORRECTO — μ y σ solo del train
X_mean = X_train.mean(axis=0)
X_std  = X_train.std(axis=0)
X_train_n = (X_train - X_mean) / X_std
X_test_n  = (X_test  - X_mean) / X_std  # mismos parámetros

# INCORRECTO — data leakage
X_mean = X.mean(axis=0)  # usa test para calcular μ → información filtrada
```
