from itertools import combinations
from pandas import DataFrame
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Leer el dataset
df = fetch_california_housing(as_frame=True).frame

# Definir variables independientes y dependientes
x_columns = df.drop("MedHouseVal", axis=1).columns.tolist()
y = df["MedHouseVal"]  # Columna objetivo

# Almacenar resultados en una lista
results = []

# Generar combinaciones de variables
for r in range(
    1, len(x_columns) + 1
):  # Cambia el rango según el número de variables que quieras combinar
    for combo in combinations(x_columns, r):
        x = df[list(combo)]  # Seleccionar las columnas de la combinación
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=0
        )

        # Crear y entrenar el modelo de regresión lineal
        model = LinearRegression()
        model.fit(x_train, y_train)

        # Evaluar el modelo
        score = model.score(x_test, y_test)

        # Almacenar los resultados
        results.append({"variables": combo, "score": score})

# Ordenar resultados de mayor a menor score
sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

# Mostrar resultados
for result in sorted_results:
    print("Variables:", result["variables"], "| Score:", result["score"])

print(
    "Highest Score is: ",
    sorted_results[0].Score,
    "with cols: ",
    sorted_results[0].Score,
)
