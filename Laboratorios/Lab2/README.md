# 🎮 Regresión Lineal Multivariable — Backloggd Games

Predicción del número de jugadores (`Plays`) de un videojuego usando tres métodos de regresión lineal implementados desde cero con NumPy.

---

## 📋 Descripción

Este proyecto implementa y compara tres modelos de regresión lineal sobre el dataset público **Backloggd Games**, que contiene métricas reales de miles de videojuegos (calificaciones, engagement, géneros, plataformas).

| Modelo | Método |
|--------|--------|
| **Gradiente Descendente** | Optimización iterativa con normalización z-score |
| **Regresión Polinómica** | Agrega términos cuadráticos (grado 2) para capturar no-linealidades |
| **Ecuación Normal** | Solución analítica exacta: θ = (XᵀX)⁻¹Xᵀy |

---

## 📁 Archivos

```
├── regresion_games_v2.ipynb   # Cuadernillo principal (3 modelos completos)
├── backloggd_games.csv        # Dataset (subir a Google Drive)
└── README.md
```

---

## 📊 Dataset

- **Fuente:** [Backloggd Games Dataset](https://www.backloggd.com/)
- **Muestras:** m > 10,000
- **Features:** n = 19

| # | Feature | Descripción |
|---|---------|-------------|
| 1 | `Rating` | Calificación promedio (0–5) |
| 2 | `Playing` | Usuarios actualmente jugando |
| 3 | `Backlogs` | Juego pendiente en lista |
| 4 | `Wishlist` | Usuarios que desean el juego |
| 5 | `Lists` | Apariciones en listas de usuarios |
| 6 | `Reviews` | Número de reseñas |
| 7 | `Release Year` | Año de lanzamiento |
| 8 | `Platforms` | Nº de plataformas disponibles |
| 9 | `Num Genres` | Cantidad de géneros |
| 10–19 | `Genre flags` | One-hot: RPG, Action, Adventure, Shooter, Strategy, Simulation, Sports, Puzzle, Platform, Fighting |

> Los valores en notación `K`/`M` (ej: `21K`, `1.5M`) son convertidos automáticamente a float.  
> Se aplica `log1p` al target para estabilizar la distribución sesgada.

---

## 🧠 Modelos

### 1. Gradiente Descendente Multivariable

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

$$\theta := \theta - \frac{\alpha}{m} X^T(X\theta - y)$$

- **α = 0.3** — mayor velocidad sin divergencia  
- **1500 iteraciones** — convergencia completa

### 2. Regresión Polinómica (grado 2)

$$\phi(x) = [x_1, x_2, \ldots, x_{19},\ x_1^2, x_2^2, \ldots, x_6^2]$$

25 features totales. Los términos cuadráticos se re-normalizan para evitar overflow.

### 3. Ecuación Normal

$$\theta = (X^TX)^{-1}X^Ty$$

Solución exacta en un paso. Usa pseudoinversa (`pinv`) para estabilidad numérica.

---

## 📈 Evaluación

Los tres modelos se validan sobre **100 predicciones aleatorias**:

| Métrica | Descripción |
|---------|-------------|
| **MAE** | Error absoluto promedio en jugadores |
| **RMSE** | Penaliza errores grandes |
| **R²** | Proporción de varianza explicada (1.0 = perfecto) |

---

## 🚀 Uso en Google Colab

1. Subir `backloggd_games.csv` a Google Drive
2. Abrir `regresion_games_v2.ipynb` en Colab
3. Ajustar `CSV_PATH` en la celda 1:
```python
CSV_PATH = '/content/gdrive/MyDrive/ruta/backloggd_games.csv'
```
4. Ejecutar todo: `Ctrl + F9`

### Predicción de un juego nuevo

Modificar el vector en la celda 10:
```python
# [Rating, Playing, Backlogs, Wishlist, Lists, Reviews,
#  Year, Platforms, Num_Genres, RPG, Adventure, Shooter,
#  Action, Strategy, Simulation, Sports, Puzzle, Platform, Fighting]
x_nuevo = np.array([4.5, 3000, 5000, 4000, 200, 500,
                    2023, 3, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
```

---

## 🔧 Dependencias

```
numpy
matplotlib
google-colab
```

---

## 📝 Estructura del Cuadernillo

| Celda | Contenido |
|-------|-----------|
| 1 | Imports y montaje de Google Drive |
| 2 | Carga del CSV, parseo K/M, log1p al target |
| 3 | Normalización z-score, construcción de X_b y X_poly_b |
| 4 | Funciones: `costo()`, `gradiente()`, `ec_normal()` |
| 5 | Modelo 1 — comparación de α, entrenamiento, curva de costo |
| 6 | Modelo 2 — regresión polinómica, curva de costo |
| 7 | Modelo 3 — ecuación normal, gráfica comparativa 3 costos |
| 8 | 100 predicciones: tabla + métricas MAE/RMSE/R² |
| 9 | Gráficas Real vs Predicho + error absoluto por muestra |
| 10 | Predicción de un juego nuevo con los 3 modelos |
| 11 | Conclusiones y tabla comparativa |
