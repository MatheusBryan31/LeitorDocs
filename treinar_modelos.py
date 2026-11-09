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
    X = X.select_dtypes(include="number")  # Mantém apenas colunas numéricas

    if X.isna().any().any():
        colunas_com_nan = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Há valores ausentes nas colunas: {colunas_com_nan}")
    print(f"Imagens: {len(X)} | Grupos (fotos originais): {grupos.nunique()}")
    print(f"Features usadas ({X.shape[1]}): {', '.join(X.columns)}")
    print("\nImagens por classe:")
    print(y.value_counts().to_string())

    return X, y, grupos

# ===================================================
# FUNÇÃO PARA DIVIDIR TREINO E TESTE (POR GRUPO)
# ===================================================
def dividir_treino_teste(X, y, grupos):
    divisor = StratifiedGroupKFold(
        n_splits=N_DOBRAS, shuffle=True, random_state=RANDOM_STATE
    )

    # ??
    indices_treino, indices_teste = next(divisor.split(X, y, grupos))
    X_treino, X_teste = X.iloc[indices_treino], X.iloc[indices_teste]
    y_treino, y_teste = y.iloc[indices_treino], y.iloc[indices_teste]
    g_treino, g_teste = grupos.iloc[indices_treino], grupos.iloc[indices_teste]

    # ALERTA: Nenhuma foto original pode estar nos dois conjuntos (treino e teste)
    repetidos = set(g_treino) & set(g_teste)
    assert not repetidos, f"Vazamento de dados! Grupos nos dois conjuntos: {repetidos}"

    print(
        f"\nTreino: {len(X_treino)} imagens ({g_treino.nunique()} grupos) | "
        f"Teste: {len(X_teste)} imagens ({g_teste.nunique()} grupos)"
    )
    print("Teste por classe:")
    print(y_teste.value_counts().to_string())

    return X_treino, X_teste, y_treino, y_teste, g_treino

# ===================================================
# FUNÇÃO PARA SALVAR A MATRIZ DE CONFUSÃO
# ===================================================
def salvar_matriz_confusao(id_modelo, nome, y_real, y_previsto):
    figura, eixo = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_real,
        y_previsto,
        labels = CLASSES,
        cmap = "Greens",
        ax = eixo,
    )
    eixo.set_title(f"Matriz de confusão - {nome}")
    eixo.set_xlabel("Classe prevista")
    eixo.set_ylabel("Classe real")
    figura.tight_layout()
    caminho = PASTA_RESULTADOS / f"matriz_{id_modelo}.png"
    figura.savefig(caminho, dpi=150)
    plt.close(figura)

# ===================================================
# FUNÇÃO PARA TREINAR E AVALIAR UM MODELO
# ===================================================
def treinar_e_avaliar(id_modelo, nome, estimador, grade, dados):
    X_treino, X_teste, y_treino, y_teste, g_treino = dados

    print("\n===================================================")
    print(f" {nome.upper()}")
    print("===================================================")


    # Pipeline: o escalonador é ajustado com os dados de cada dobra de treino
    pipeline = Pipeline([
        ("escala", StandardScaler()),
        ("modelo", estimador),
    ])

    validacao = StratifiedGroupKFold(
        n_splits = N_DOBRAS, shuffle=True, random_state = RANDOM_STATE
    )

    busca = GridSearchCV(
        pipeline,
        param_grid = grade,
        scoring = "f1_macro",
        cv = validacao,
        n_jobs = -1,
    )
    busca.fit(X_treino, y_treino, groups = g_treino)

    melhor = busca.best_estimator_
    print(f"Melhores parâmetros: {busca.best_params_}")
    print(f"F1 macro na validação cruzada: {busca.best_score_:.4f}")

    # Avaliação final no conjunto de teste (dados que o modelo não viu)
    previsto = melhor.predict(X_teste)

    print("\nRelatório pro classe (conjunto de teste):")
    print(classification_report(
        y_teste, previsto, labels=CLASSES, digits=4, zero_division=0
    ))

    matriz = confusion_matrix(y_teste, previsto, labels = CLASSES)
    print("Matriz de confusão (linhas = real, colunas = previsto):")
    print(pd.DataFrame(
        matriz,
        index = [f"real_{c}" for c in CLASSES],
        columns = [f"prev_{c}" for c in CLASSES],
    ).to_string())

    salvar_matriz_confusao(id_modelo, nome, y_teste, previsto)

    recalls = recall_score(
        y_teste, previsto, labels = CLASSES, average = None, zero_division = 0
    )

    resultado = {
        "modelo": nome,
        "acuracia": accuracy_score(y_teste, previsto),
        "precision_macro": precision_score(
            y_teste, previsto, labels = CLASSES, average = "macro", zero_division = 0
        ),
        "recall_macro": recall_score(
            y_teste, previsto, labels = CLASSES, average = "macro", zero_division = 0
        ),
        "f1_macro": f1_score(
            y_teste, previsto, labels = CLASSES, average = "macro", zero_division = 0
        ),
        "recall_minimo": float(recalls.min()),
    }
    for classe, valor in zip(CLASSES, recalls):
        resultado[f"recall_{classe}"] = float(valor)
    resultado["atende_recall_minimo"] = bool(recalls.min() >= RECALL_MINIMO)
    resultado["parametros"] = str(busca.best_params_)

    return resultado, melhor

# ===================================================
# ???
# ===================================================
def mostrar_importancias(pipeline, nomes_features):
    modelo = pipeline.named_steps["modelo"]
    if not hasattr(modelo, "feature_importances"):
        return

    importancias = pd.Series(modelo.feature_importances_, index = nomes_features)
    print("\nFeatures mais importantes do modelo escolhido:")
    print(importancias.sort_values(ascending = False).head(10).round(4).to_string())

# ===================================================
# FUNÇÃO PRINCIPAL
# ===================================================
def main():
    print("# ===================================================")
    print(" TREINAMENTO - PROJETO ==> LeitorDocs")
    print("# ===================================================")

    PASTA_RESULTADOS.mkdir(exist_ok = True)
    PASTA_MODELOS.mkdir(exist_ok = True)

    X, y, grupos = carregar_dados()
    dados = dividir_treino_teste(X, y, grupos)

    resultados = []
    modelos_treinados = {}

    for id_modelo, nomme, estimador, grade in montar_modelos():
        resultado, melhor = treinar_e_avaliar(id_modelo, nome, estimador, grade, dados)
        resultados.append(resultado)
        modelos_treinados[nome] = melhor

        # Ranking: 1º recall mínimo entre as classes, 2º F1 macro, 3º acurácia
        tabela = pd.DataFrame(resultados).sort_values(
            by = ["recall_minimo", "f1_macro", "acuracia"], ascending = False
        )
        tabela.to_csv(PASTA_RESULTADOS / "comparacao_modelos.csv", index = False)

        colunas_exibir = [
            "modelo", "acuracia", "precision_macro", "recall_macro", "f1_macro",
            "recall_minimo", "atende_recall_minimo",
        ]
        pd.set_option("display.width", 200)
        pd.set_optin("display.max_columns", None)
        print("\n===================================================")
        print(" COMPARAÇÃO DOS MODELOS (conjunto de teste)")
        print("===================================================")
        print(tabela[colunas_exibir].round(4).to_string(index = False))

        melhor_nome = tabela.iloc[0]["modelo"]
        melhor_modelo = modelos_treinados[melhor_nome]
        print(f"\nMelhor modelo: {melhor_nome}")

        if tabela.iloc[0]["atende_recall_minimo"]:
            print(f"Atende ao recall mínimo de {RECALL_MINIMO: .0%} em todas as classes.")
        else:
            print(f"Não atinge o recall mínimo de {RECALL_MINIMO: .0%} em todas as classes.")

        mostrar_importancias(melhor_modelo, list(X.columns))

        caminho_modelo = PASTA_MODELOS / "modelo_final.joblib"
        joblib.dump(
            {
                "pipeline": melhor_modelo,
                "features": list(X.columns),
                "classes": CLASSES,
            },
            caminho_modelo,
        )
        print(f"\nModelo salvo em: {caminho_modelo}")
        print(f"Resultados salvos em: {PASTA_RESULTADOS}")

# ===================================================
# EXECUÇÃO DO PROGRAMA
# ===================================================
if __name__ == "__main__":
    main()