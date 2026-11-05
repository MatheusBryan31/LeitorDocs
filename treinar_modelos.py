from pathlib import Path
import joblib
import matplotlib

# Backend "Agg": salva figuras em arquivo, sem abrir janelas.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GrandientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ===================================================
# CONFIGURAÇÕES
# ===================================================
# Diretório principal do projeto
BASE_DIR = Path(__file__).resolve().parent

# Tabela gerada pelo extrator_features.py
ARQUIVO_FEATURES = BASE_DIR / "features.csv"

# Pastas de saída
PASTA_RESULTADOS = BASE_DIR / "resultados"
PASTA_MODELOS = BASE_DIR / "modelos"

# Classes do problema (a ordem define a ordem das matrizes de confusão)
CLASSES = ["nascimento", "casamento", "outros"]

# Semente aleatória: garante que os resultados sejam reproduzíveis
RANDOM_STATE = 42

# Número de dobras. Também define o teste: 1 dobra de 5 = 20% dos grupos
N_DOBRAS = 5

# Exigência do trabalho: recall mínimo em CASA classe
RECALL_MINIMO = 0.97

# Colunas que não são features (não devem ser usadas para treinar o modelo)
COLUNAS_NAO_FEATURES = ["arquivo", "classe", "grupo", "variacao", "ocr_texto"]

# Features que você quer desligar para testar (ex.: ["largura, altura, proporcao"])
COLUNAS_IGNORADAS = []

# ===================================================
# MODELOS E PARÂMETROS TESTADOS
# ===================================================
def montar_modelos():
    # Cada item: (id do arquivo, nome de exibição, modelo, grade de parâmetros)
    # O prefixo "modelo__" indica que o parâmetro pertence ao passo "modelo" do Pipeline
    return[
        (
            "regressao_logistica",
            "Regressão Logística",
            LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
            {
                "modelo__C": [0.1, 1, 10, 100],
                "modelo__class_wight": [None, "balanced"],
            },
        ),
        (
            "svm",
            "SVM (kernel RBF)",
            SVC(kernel="rbf", random_state=RANDOM_STATE),
            {
                "modelo__C": [0.1, 1, 10, 100],
                "modelo__gamma": ["scale", 0.01, 0.1],
                "modelo__class_weight": [None, "balanced"],
            },
        ),
        (
            "random_forest",
            "Random Forest",
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "modelo__n_estimators": [100, 300],
                "modelo__max_depth": [None, 5, 10],
                "modelo__min_samples_leaf": [1, 3],
            },
        ),
        (
            "gradient_boosting",
            "Gradient Boosting",
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "modelo__n_estimators": [100, 200],
                "modelo__learning_rate": [0.05, 0.1],
                "modelo__max_depth": [2, 3],
            },
        ),
    ]

# ===================================================
# FUNÇÃO PARA CARREGAR OS DADOS
# ===================================================
def carregar_dados():
    if not ARQUIVO_FEATURES.exists():
        raise FileNotFoundError(
            f"{ARQUIVO_FEATURES.name} não encontrado. Rode o extrator_features.py antes."
        )
    tabela = pd.read_csv(ARQUIVO_FEATURES)
    y = tabela["classe"]
    grupos = tabela["grupo"]

    # Features = ??
    X = tabela.drop(columns=COLUNAS_NAO_FEATURES + COLUNAS_IGNORADAS, errors="ignore")
    