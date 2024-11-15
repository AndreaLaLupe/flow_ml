"""_summary_: este código genera los modelos de clasificación para el problema de titanic
"""

import pickle
import pandas as pd

# sklearn tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def train_model():
    """_summary_: esta función entrena los modelos de clasificación y almacena el mejor
    """

    # Cargamos los datos
    dataset = pd.read_csv('../data/processed/features_for_model.csv')
    dataset.head()

    # Seleccionar target y features
    x = dataset.drop(['Survived'], axis=1)
    y = dataset['Survived']

    # Split train y test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, 
        shuffle=True, random_state=2025)

    # Configuramos scaler
    std_scaler = StandardScaler()
    std_scaler.fit(x_train) # Calculamos valores de entrenamiento

    with open('../artifacts/std_scaler.pkl', 'wb') as f:
        pickle.dump(std_scaler, f)

    # Creamos modelos de predicción
    x_train_std = std_scaler.transform(x_train)

    # Random Forest
    model_rf = RandomForestClassifier()
    model_rf.fit(x_train_std, y_train)

    x_test_std = std_scaler.transform(x_test)
    y_preds_rf = model_rf.predict(x_test_std)
    rf_acc = accuracy_score(y_test, y_preds_rf)

    ## Regresión logística
    model_rl = LogisticRegression()
    model_rl.fit(x_train_std, y_train)

    y_preds_rl = model_rl.predict(x_test_std)
    rl_acc = accuracy_score(y_test, y_preds_rl)

    if rf_acc>=rl_acc:
        winner_model = model_rf
    else:
        winner_model = model_rl

    with open('../models/titanic_model_v1.pkl', 'wb') as f:
        pickle.dump(winner_model, f)