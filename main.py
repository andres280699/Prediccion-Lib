import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.express as px

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

# Crear un DataFrame para los resultados reales y predichos
results = pd.DataFrame({'True Rating': y_test, 'Predicted Rating': y_pred})

<<<<<<< HEAD
# Agregar la información de los libros a los resultados
results = pd.concat([results, X_test], axis=1)

# Seleccionar las características relevantes para la visualización
hover_features = ['bookID', 'title', 'authors', 'publisher', 'ratings_count', 'text_reviews_count']

# Graficar utilizando Plotly
fig = px.scatter(results, x='True Rating', y='Predicted Rating', hover_data=hover_features, text='bookID')

# Configurar el diseño de la gráfica
fig.update_layout(
    title="Predicción del rating de libros",
    xaxis_title="Valor real",
    yaxis_title="Predicción",
    template="plotly_dark"
)

# Mostrar el gráfico interactivo
fig.show()
=======
# Agregar etiquetas a los puntos en la gráfica
for i in range(len(X_test)):
    #las siguientes lineas de codigo son los textos que apareceran en la grafica
#    plt.annotate(f"{X_test.iloc[i]['bookID']}: {X_test.iloc[i]['title']} - {X_test.iloc[i]['authors']} - {X_test.iloc[i]['publisher']}",
#                 (y_test.iloc[i], y_pred[i]))
    plt.annotate(f"{X_test.iloc[i]['title']}",
                 (y_test.iloc[i], y_pred[i]))
plt.show()
>>>>>>> f19ad1a2f6856c9b2cef19692e5392c8913a26b5
