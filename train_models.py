import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import os

# ======================== Пути ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)
data_dir = os.path.join(BASE_DIR, "upload")
dataset_path = os.path.join(data_dir, "diamonds_processed.csv")

# ======================== Загрузка данных ========================
df = pd.read_csv(dataset_path, sep=';')

# Ограничим выборку (опционально)
N_SAMPLES = 20000
df = df.iloc[:N_SAMPLES]

X = df.drop(['price'], axis=1)
y = df['price']

# ======================== Предобработка ========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Сохраняем скейлер
scaler_path = os.path.join(models_dir, "scaler.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

results = {}

# ======================== ML1: Linear Regression ========================
ml1 = LinearRegression()
ml1.fit(X_train, y_train)
results['ML1'] = r2_score(y_test, ml1.predict(X_test))
with open(os.path.join(models_dir, "ml1.pkl"), 'wb') as f:
    pickle.dump(ml1, f)

# ======================== ML2: Gradient Boosting ========================
ml2 = GradientBoostingRegressor(n_estimators=100, random_state=42)
ml2.fit(X_train, y_train)
results['ML2'] = r2_score(y_test, ml2.predict(X_test))
with open(os.path.join(models_dir, "ml2.pkl"), 'wb') as f:
    pickle.dump(ml2, f)

# ======================== ML3: HistGradientBoosting ========================
from sklearn.ensemble import HistGradientBoostingRegressor
ml3 = HistGradientBoostingRegressor(random_state=42)
ml3.fit(X_train, y_train)
results['ML3'] = r2_score(y_test, ml3.predict(X_test))
with open(os.path.join(models_dir, "ml3.pkl"), 'wb') as f:
    pickle.dump(ml3, f)

# ======================== ML4: Random Forest ========================
ml4 = RandomForestRegressor(n_estimators=100, random_state=42)
ml4.fit(X_train, y_train)
results['ML4'] = r2_score(y_test, ml4.predict(X_test))
with open(os.path.join(models_dir, "ml4.pkl"), 'wb') as f:
    pickle.dump(ml4, f)
# Удаляем ссылки на обучающие данные, если они остались в объектах деревьев
for estimator in ml4.estimators_:
    estimator.tree_.value.flags.writeable = True # Разрешаем правку, если нужно

# ======================== ML5: Stacking ========================
estimators = [
    ('lr', LinearRegression()),
    ('rf', RandomForestRegressor(n_estimators=10, random_state=42))
]
ml5 = StackingRegressor(estimators=estimators, final_estimator=GradientBoostingRegressor())
ml5.fit(X_train, y_train)
results['ML5'] = r2_score(y_test, ml5.predict(X_test))
with open(os.path.join(models_dir, "ml5.pkl"), 'wb') as f:
    pickle.dump(ml5, f)

# ======================== ML6: MLP Neural Network ========================
ml6 = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
ml6.fit(X_train, y_train)
results['ML6'] = r2_score(y_test, ml6.predict(X_test))
with open(os.path.join(models_dir, "ml6.pkl"), 'wb') as f:
    pickle.dump(ml6, f)

# ======================== Результаты ========================
print("Training completed. R2 scores:")
for name, score in results.items():
    print(f"{name}: {score:.4f}")