# Laboratorio 4 - Regresion Logistica Multiclase (OvR)

Este repositorio contiene dos cuadernos donde aplicamos clasificacion multiclase con enfoque One-vs-Rest (OvR) usando regresion logistica.

## Que hacemos en cada cuaderno

- `Lab4_TPS_NoGrafico.ipynb`:
  Trabajamos con el dataset Tabular Playground Series (May 2021), balanceamos clases, preprocesamos datos (split y escalado), entrenamos un modelo OvR y evaluamos su desempeno con accuracy, log-loss, matriz de confusion, curvas ROC y curvas de aprendizaje.

- `OvR_Logistica_Muticlase_Lab4.ipynb`:
  Trabajamos con Spoken Arabic Digit (MFCC), implementamos y explicamos la parte matematica de regresion logistica regularizada y el esquema One-vs-All, entrenamos clasificadores por clase y evaluamos con precision/recall/F1, matriz de confusion y analisis de probabilidades por ejemplo.

## Idea general

En ambos casos el objetivo es entender el flujo completo de un problema de clasificacion multiclase: preparar datos, entrenar con OvR y analizar resultados para interpretar el comportamiento del modelo.
