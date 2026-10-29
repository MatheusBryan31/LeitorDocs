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