from PIL import Image, ImageEnhance
from pathlib import Path

# ===================================
# CONFIGURAÇÕES
# ===================================

# Diretório principal do projeto
BASE_DIR = Path(__file__).resolve().parent

# Pasta principal das imagens
PASTA_BASE = BASE_DIR / "src/assets/imagens"

# Classes que serão processadas
CLASSES = ["nascimento", "casamento", "outros"]

# Extensões permitidas
EXTENSOES = [".jpg", ".jpeg", ".png"]

# ===================================
# FUNÇÃO PARA ROTACIONAR IMAGENS
# ===================================

def rotacionar(imagem, angulo):
    return imagem.rotate(
        angulo,
        expand = True,
        resample = Image.Resampling.BICUBIC,
        fillcolor = "white"
    )

# ===================================
# FUNÇÃO PARA ALTERAR O BRILHO
# ===================================

def alterar_brilho(imagem, fator):
    ajuste = ImageEnhance.Brightness(imagem)
    return ajuste.enhance(fator)

# ===================================
# FUNÇÃO PARA SALVAR IMAGENS
# ===================================

def salvar_imagem(imagem, pasta_destino, classe, contador, tipo):
    nome_arquivo = f"{classe}_{contador}_{tipo}.jpg"
    caminho = pasta_destino / nome_arquivo
    imagem.save(caminho, "JPEG", quality = 95)
    return caminho

# ===================================
# FUNÇÃO PARA PROCESSAR UMA CLASSE
# ===================================

def processar_classe(classe):
    print(f"\nProcessando classe: {classe.upper()}")

    # Define as pastas automaticamente
    pasta_origem = PASTA_BASE / classe / f"{classe}Origem"
    pasta_destino = PASTA_BASE / classe / f"{classe}Destino"

    # Verifica se a pasta de origem existe
    if not pasta_origem.exists():
        print(f"Pasta de origem não encontrada: {pasta_origem}")
        return 0

    # Cria a pasta de destino
    pasta_destino.mkdir(parents = True, exists_ok = True)

    # Busca os arquivos de imagem
    arquivos = sorted([
        arquivo
        for arquivo in pasta_origem.iterdir()
        if arquivo.is_file()
        and arquivo.suffix.lower() in EXTENSOES
    ])

    # Verifica se existem imagens
    if not arquivos:
        print("Nenhuma imagem encontrada.")
        return 0
    contador = 1
    total_gerado = 0

    # Percorre todas as imagens
    for arquivo_imagem in arquivos:
        try:
            with Image.open(arquivo_imagem) as imagem:
                
                # Converte para RGB
                original = imagem.convert("RGB")

                # ===================================
                # IMAGEM ORIGINAL
                # ===================================

                salvar_imagem(
                    original,
                    pasta_destino,
                    classe,
                    contador,
                    "original"
                )
                total_gerado += 1

                # ===================================
                # ROTAÇÃO +8 GRAUS
                # ===================================

                rotacionada_esquerda = rotacionar(original, 8)
                salvar_imagem(
                    rotacionada_esquerda,
                    pasta_destino,
                    classe,
                    contador,
                    "rotacao_positiva"
                )
                total_gerado += 1

                # ===================================
                # ROTAÇÃO -8 GRAUS
                # ===================================

                rotacionada_direita = rotacionar(original, -8)
                salvar_imagem(
                    rotacionada_direita,
                    pasta_destino,
                    classe,
                    contador,
                    "rotacao_negativa"
                )
                total_gerado += 1

                # ===================================
                # BRILHO +20%
                # ===================================
                mais_clara = alterar_brilho(original, 1.2)
                salvar_imagem(
                    mais_clara,
                    pasta_destino,
                    classe,
                    contador,
                    "mais_clara"
                )
                total_gerado += 1

                # ===================================
                # BRILHO -20%
                # ===================================