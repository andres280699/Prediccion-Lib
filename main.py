import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # Agregar esta línea de importación

# Ruta de la fuente Arial Unicode MS instalada en tu sistema
font_path = "/Users/andres arturo perez/PycharmProjects/Prediccion-Lib/arial-unicode-ms.ttf"

# Cargar la fuente
fontprop = fm.FontProperties(fname=font_path)

# Configurar la fuente predeterminada para matplotlib
plt.rcParams['font.family'] = fontprop.get_name()

# Cargar los datos desde el archivo CSV
data = pd.read_csv("/Users/andres arturo perez/PycharmProjects/Prediccion-Lib/booksf.csv")

# Seleccionar las características relevantes para la predicción del rating
features = ['bookID', 'title', 'authors', 'publisher', 'average_rating', 'ratings_count', 'text_reviews_count']

# Filtrar y limpiar los datos relevantes
data = data[features].dropna()

# Convertir los valores de la columna 'average_rating' a tipo float, tratando los valores no válidos como NaN
data['average_rating'] = pd.to_numeric(data['average_rating'], errors='coerce')

# Eliminar las filas que contienen valores no válidos en la columna 'average_rating'
data = data.dropna(subset=['average_rating'])

# Separar los datos en características (X) y variable objetivo (y)
X = data.drop('average_rating', axis=1)
y = data['average_rating']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear una instancia del modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo utilizando los datos de entrenamiento
model.fit(X_train[['bookID', 'ratings_count', 'text_reviews_count']], y_train)

# Realizar predicciones utilizando los datos de prueba
y_pred = model.predict(X_test[['bookID', 'ratings_count', 'text_reviews_count']])

# Calcular el error cuadrático medio (MSE) para evaluar la precisión del modelo
mse = mean_squared_error(y_test, y_pred)
print("Error cuadrático medio (MSE):", mse)

# Graficar los valores reales y las predicciones
plt.scatter(y_test, y_pred)
plt.xlabel("Valor real")
plt.ylabel("Predicción")
plt.title("Predicción del rating de libros")

# Agregar etiquetas a los puntos en la gráfica
for i in range(len(X_test)):
    #las siguientes lineas de codigo son los textos que apareceran en la grafica
#    plt.annotate(f"{X_test.iloc[i]['bookID']}: {X_test.iloc[i]['title']} - {X_test.iloc[i]['authors']} - {X_test.iloc[i]['publisher']}",
#                 (y_test.iloc[i], y_pred[i]))
    plt.annotate(f"{X_test.iloc[i]['title']}",
                 (y_test.iloc[i], y_pred[i]))
plt.show()
