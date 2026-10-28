# LeitorDocs

Classificador de documentos com Machine Learn para automatizar a triagem de certidões no Instituto Federal de Almenara.

---

## Sumário

- [Problema](#problema)
- [Objetivo](#objetivo)
- [Classes do modelo](#classes-do-modelo)
- [Preparação do dataset](#preparação-do-dataset)
- [Tecnologias utilizadas](#tecnologias-utilizadas)
- [Instalação](#instalação)
- [Como o código funciona](#como-o-código-funciona)
- [Status do projeto](#status-do-projeto)

---

## Problema

Nos períodos de cadastro em eventos, matrículas e inscrições, o Campus Almenara ainda analisa os documentos recebidos de forma **manual**, por meio de um profissional. Esse processo consome muito tempo.

Surge, então, a necessidade de um sistema que resolva esse problema: um modelo de **Machine Learn** capaz de analisar os documentos automaticamente.

## Objetivo

Ler os documentos recebidos e distinguir, entre eles, quais são certidões de nascimento, quais são certidões de casamento e quais não se enquadram em nenhum dos dois casos.

## Classes do modelo

| Classe       | Descrição                                                                                             |
| ------------ | ----------------------------------------------------------------------------------------------------- |
| `nascimento` | Certidões de nascimento                                                                               |
| `casamento`  | Certidões de casamento                                                                                |
| `outros`     | Documentos não identificados (fotos borradas, tremidas ou escuras) ou diferentes de uma certidão      |

## Preparação do dataset

Antes da criação do modelo, foi necessário reunir e tratar os dados.

- **Dados originais:** 30 imagens de cada classe.
- **Data Augmentation:** Para cada imagem original, são geradas cinco novas imagens com ângulo e clareza diferentes, aumentando a variabilidade do dataset.

> **O que é Data Augmentation?** É a técnica de recriar uma imagem em outros ângulos e tons, ampliando o dataset sem a necessidade de coletar novos dados.

## Tecnologias utilizadas

- Python versão 3.13.17
- [Pillow](https://pypi.org/project/pillow/): biblioteca para manupulação de imagens

## Instalação

Foi criado um ambiente para as bibliotecas do Python e posteriormente foi instalado o Pillow:

```bash
pip install Pillow
```