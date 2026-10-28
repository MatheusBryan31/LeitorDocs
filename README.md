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