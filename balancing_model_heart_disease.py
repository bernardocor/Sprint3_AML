
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler, SMOTE

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

# Preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features),
    ]
)

# Oversampling
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X_train, y_train)

# SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

def train_evaluate(X_res, y_res, label):
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    model.fit(X_res, y_res)

    y_train_pred = model.predict(X_res)
    y_test_pred = model.predict(X_test)

    print(f"--- Resultados con {label} ---")
    print("Entrenamiento:")
    print(" Accuracy:", accuracy_score(y_res, y_train_pred))
    print(" Precision:", precision_score(y_res, y_train_pred))
    print(" Recall:", recall_score(y_res, y_train_pred))
    print(" F1 Score:", f1_score(y_res, y_train_pred))

    print("Prueba:")
    print(" Accuracy:", accuracy_score(y_test, y_test_pred))
    print(" Precision:", precision_score(y_test, y_test_pred))
    print(" Recall:", recall_score(y_test, y_test_pred))
    print(" F1 Score:", f1_score(y_test, y_test_pred))
    print()

train_evaluate(X_ros, y_ros, "Random OverSampling")
train_evaluate(X_smote, y_smote, "SMOTE")
