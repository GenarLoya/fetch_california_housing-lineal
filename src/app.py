from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# leer el dataset
df = fetch_california_housing()
print('--- Head ---')
print(df.head())

# separa los datos
variables_independientes = ["x", "y", "z"]
x = df[variables_independientes]  # Selección corregida
y = df['price']


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)

print('--- Split ---')
print('x_train:', x_train)
print('x_test:', x_test)
print('y_train:', y_train)
print('y_test:', y_test)

# entrenar el modelo
model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print('--- Prediction ---')
print('y_pred:', y_pred)

# predecir
pr = model.predict([[0.21, 0.21]])
print('--- Predict ---')
print('pr:', pr)

# score
score = model.score(x_test, y_test)
print('--- Score ---')
print('score:', score)

# sns plot
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharey=True)
ax[0].scatter(x_train['x'], y_train)
ax[0].plot(x_train['x'], model.predict(x_train), c='g')
ax[0].set_title('Carats vs Price (training)')
ax[0].set_xlabel('Width')
ax[0].set_ylabel('Price USD')
ax[1].scatter(x_test['x'], y_test)
ax[1].plot(x_test['x'], model.predict(x_test), c='g')
ax[1].set_title('Carats vs Price (testing)')
ax[1].set_xlabel('Width')
ax[1].set_ylabel('Price USD')
plt.suptitle('Linear Regression Model')

# from sklearn.neural_network import MLPClassifier
# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn import metrics
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import KBinsDiscretizer


# X, y = df.data, df.target
# discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
# y_binned = discretizer.fit_transform(y.reshape(-1, 1)).ravel()


# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_binned, test_size=0.2, random_state=1)

# # Estadrizacion de los datos eliminando la media y escalando de forma que la varianza sea igua a 1
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# mlp = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32),
#                     max_iter=2000, random_state=1, verbose=True, activation='relu')
# mlp.fit(X_train, y_train)

# y_pred = mlp.predict(X_test)
# print(y_pred)

# accuracy = accuracy_score(y_test, y_pred)

# print(accuracy)

# conf_matrix = confusion_matrix(y_test, y_pred)

# # Draw confusion matrix
# for i in range(len(conf_matrix)):
#     print(conf_matrix[i])
