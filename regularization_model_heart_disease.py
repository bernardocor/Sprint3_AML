
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Cargar datos
dtype_dict = {
    "HeartDisease": "category", "BMI": "float64", "Smoking": "category",
    "AlcoholDrinking": "category", "Stroke": "category", "PhysicalHealth": "float64",
    "MentalHealth": "float64", "DiffWalking": "category", "Sex": "category",
    "AgeCategory": "category", "Race": "category", "Diabetic": "category",
    "PhysicalActivity": "category", "GenHealth": "category", "SleepTime": "float64",
    "Asthma": "category", "KidneyDisease": "category", "SkinCancer": "category"
}

df = pd.read_csv("processing_heart_disease.csv", dtype=dtype_dict)
X = df.drop(columns="HeartDisease")
y = df["HeartDisease"].astype("category").cat.codes

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Identificar columnas
num_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
cat_features = X.select_dtypes(include=["category", "object"]).columns.tolist()

# Pipeline de preprocesamiento y modelo
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features),
    ]
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Búsqueda de hiperparámetro C
param_grid = {
    "classifier__C": [1, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.000001]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Evaluación
y_train_pred = grid_search.predict(X_train)
y_test_pred = grid_search.predict(X_test)

print("--- Evaluación en conjunto de entrenamiento ---")
print("Accuracy:", accuracy_score(y_train, y_train_pred))
print("Precision:", precision_score(y_train, y_train_pred))
print("Recall:", recall_score(y_train, y_train_pred))
print("F1 Score:", f1_score(y_train, y_train_pred))

print("--- Evaluación en conjunto de prueba ---")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred))
print("Recall:", recall_score(y_test, y_test_pred))
print("F1 Score:", f1_score(y_test, y_test_pred))

print("Mejor valor de C:", grid_search.best_params_["classifier__C"])
