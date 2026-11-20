# Modelo de propensión de compra

Prueba técnica · Data Scientist · Ferreycorp — Brenda Sarita Hilario Romero

Predice, para cada visita de un cliente, la probabilidad de que compre cada una de las 5 marcas de la categoría. La probabilidad se descompone en dos modelos:

```
P(compra marca k) = P(compra) × P(marca k | compra)
```

- **Modelo 1 · Incidencia:** ¿comprará en esta visita? LightGBM binario, monótono en precio y promoción.
- **Modelo 2 · Elección de marca:** si compra, ¿qué marca elige? LightGBM multiclase.

## Resultados en prueba (días 641–730, nunca usados para decidir)

| Modelo | Métrica | Línea base | Modelo |
|---|---|---|---|
| 1 · ¿comprará? | PR-AUC | 0,436 | **0,474** |
| 1 · ¿comprará? | ROC-AUC | 0,666 | **0,711** |
| 2 · ¿qué marca? | Log-loss | 0,749 | **0,693** |

- Las líneas base son la tasa de compra histórica del cliente (modelo 1) y su cuota histórica de marcas (modelo 2).
- El 20 % de las visitas con mayor propensión concentra el **40,9 % de las compras**.
- El precio de la categoría es el tercer factor que más mueve la compra (SHAP).

## Metodología

- **Partición por fecha:** entrenamiento (días 1–550), validación (551–640) y prueba (641–730).
- **Variables de historial** construidas solo con visitas anteriores, verificadas con dos pruebas automáticas de no fuga.
- **Selección en validación** entre línea base, regresión logística y LightGBM, con ajuste de hiperparámetros.
- **Evaluación única en prueba** y explicabilidad con SHAP.

## Cómo ejecutar

1. Instalar dependencias (probado con Python 3.13):

   ```bash
   pip install pandas==3.0.5 scikit-learn==1.9.1 lightgbm==4.7.0 shap==0.54.1 matplotlib seaborn jupyter
   ```

2. Colocar `compras_data.csv` en la misma carpeta que el notebook.
3. Abrir `Ferreycorp.ipynb` y ejecutar todas las celdas en orden. Tarda menos de un minuto en CPU.

## Estructura del notebook

| Paso | Contenido |
|---|---|
| 1 | Definición del problema y de las targets |
| 2 | Auditoría de datos: 11 verificaciones de integridad |
| 3 | Limpieza y partición temporal |
| 4 | EDA en entrenamiento |
| 5 | Variables, pruebas de no fuga, segmentación (K-means) y preprocesamiento |
| 6 | Modelado, ajuste, evaluación en prueba y explicabilidad |
