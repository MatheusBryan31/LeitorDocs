import re
import unicodedata
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytesseract
from pytesseract import Output

# ===================================
# CONFIGURAÇÕES
# ===================================
# Diretório principal do projeto
BASE_DIR = Path(__file__).resolve().parent

# Pasta principal das imagens
PASTA_BASE = BASE_DIR / "src/assets/imagens"

# Arquivo CSV que receberá a tabela de features
ARQUIVO_SAIDA = BASE_DIR / "features.csv"

# Classes que será processadas
CLASSES = ["nascimento", "casamento", "outros"]

# Extensões permitidas
EXTENSOES = [".jpg", ".jpeg", ".png"]

# Caminho padrão do Tesseract no Windows
CAMINHO_TESSERACT = r"C:\ProgamFiles\Tesseract-OCR\tesseract.exe"

# Idioma do OCR:
IDIOMA_OCR = "por"

# Todas as imagens são redimencionadas para esta largura antes da análise,
# para que as medidas sejam comparáveis entre fotos de tamanhos diferentes
LARGUR_PADRAO = 1600

# Tempo máximo(segundos) que o OCR pode gastar em uma única imagem
TIMEOUT_OCR = 90

if Path(CAMINHO_TESSERACT).exists():
    pytesseract.pytesseract.tesseract_cmd = CAMINHO_TESSERACT

# ===================================
# PALAVRAS-CHAVE
# ===================================
PALAVRAS_NASCIMENTO = [
    "nascimento", "nascido", "nascida", "filiacao", "avos", "gemeo",
]
PALAVRAS_CASAMENTO = [
    "casamento", "casado", "casada", "conjugue", "conjuges", "regime de bens", "nubente", "nubentes",
]
PALAVRAS_CERTIDAO = [
    "certidao", "registro civil", "matricula", "cartorio", "oficial de registro", "republica federativa",
]
FRASE_NASCIMENTO = "certidao de nascimento"
FRASE_CASAMENTO = "certidao de casamento"

# ===================================
# FUNÇÃO PARA LER UMA IMAGEM
# ===================================
def ler_imagem(caminho):
    dados = np.fromfile(str(caminho), dtype=np.uint8)
    return cv2.imdecode(dados, cv2.IMREAD_COLOR)

# ===================================
# FUNÇÃO PARA PADRONIZAR A LARGURA
# ===================================
def padronizar_largura(imagem, largura_alvo):
    altura, largura = imagem.shape[:2]
    if largura == largura_alvo:
        return imagem
    escala = largura_alvo / largura
    nova_altura = int(altura * escala)

    # INTER_AREA é o melhor para reduzir; INTER_CUBIC é o melhor para ampliar
    interpolacao = cv2.INTER_AREA if escala < 1 else cv2.INTER_CUBIC
    return cv2.resize(imagem, (largura_alvo, nova_altura), interpolation=interpolacao)

# ===================================
# FUNÇÃO PARA NORMALIZAR TEXTO
# ===================================
def normalizar_texto(texto):
    # Separa as letras dos acentor ("ã" vira "a" + "~") e descarta os acentos
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.lower()

    # Troca qualquer sequências de espaços/quebras de linha por um único espaço
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

# ===================================
# FEATURES VISUAIS (qualidade e aparência)
# ===================================
def extrair_features_visuais(original, padronizada):
    altura, largura = original.shape[:2]
    cinza = cv2.cvtColor(padronizada, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(padronizada, cv2.COLOR_BGR2HSV)

    # Bordas detectadas pelo algoritmo de Canny (imagem preta com bordas brancas)
    bordas = cv2.Canny(cinza, 100, 200)

    # Binarização de Otsu: separa "tinta" (escuro) do "papel" (claro)
    _, tinta = cv2.threshold(
        cinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return {
        "largura": largura,
        "altura": altura,
        "proporcao": round(largura/altura, 4),
        "brilho_medio": float(cinza.mean()),
        "contraste": float(cinza.std()),
        "nitidez": float(cv2.Laplacin(cinza, cv2.CV_64F).var()),
        "densidade_bordas": float(np.count_nonzero(bordas)/ bordas.size),
        "prop_escuros": float(np.mean(cinza < 50)),
        "prop_claros": float(np.mean(cinza > 200)),
        "saturacao_media": float(hsv[:, :, 1].mean()),
        "prop_tinta": float(np.count_nonzero(tinta) / tinta.size),
    }

# ===================================
# OCR: LÊ O TEXTO E A CONFIANÇA DE CADA PALAVRA
# ===================================
def executar_ocr(padronizada):
    cinza = cv2.cvtColor(padronizada, cv2.COLOR_BGR2GRAY)

    dados = pytesseract.image_to_data(
        cinza,
        lang=IDIOMA_OCR,
        output_type=Output.DICT,
        timeout=TIMEOUT_OCR,
    )

    palavras = []
    confiancas = []

    for texto, conf in zip(dados["text"],dados["conf"]):
        texto = texto.strip()
        conf = float(conf)

        # conf = -1 indica que o bloco não é uma palavra (é só estrutura da página)
        if texto and conf >= 0:
            palavras.append(texto)
            confiancas.append(conf)
    return " ".join(palavras), confiancas

# ===================================
# FUNÇÃO PARA CONTAR PALAVRAS CHAVE
# ===================================
def contar_palavras_chave(texto, palavras):
    return sum(texto.count(palavra) for palavra in palavras)

# ===================================
# FEATURES DE TEXTO (a partir do OCR)
# ===================================
def extrair_features_texto(texto_bruto, confiancas):
    texto = normalizar_texto(texto_bruto)
    sem_espacos = texto.replace(" ", "")
    total_caracteres = len(sem_espacos)
    total_letras = sum(c.isalpha() for c in sem_espaco)

    return {
        "ocr_n_caracteres": total_caracteres,
        "ocr_n_palavras": len(confiancas),
        "ocr_conf_media": float(np.mean(confiancas)) if confiancas else 0.0,
        "ocr_prop_letras": total_letras / total_caracteres if total_caracteres else 0.0,
        "kw_nascimento": contar_palavras_chave(texto, PALAVRAS_NASCIMENTO),
        "kw_casamento": contar_palavras_chave(texto, PALAVRAS_CASAMENTO),
        "kw_cetirdao": contar_palavras_chave(texto, PALAVRAS_CERTIDAO),
        "tem_frase_nascimento": int(FRASE_NASCIMENTO in texto),
        "tem_frase_casamento": int(FRASE_CASAMENTO in texto),

        "ocr_texto": texto,
    }