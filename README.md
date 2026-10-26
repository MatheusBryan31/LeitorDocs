# LeitorDocs

###### ===================================
###### PROBLEMA
###### ===================================
O IF de Almenara, em período de cadastro de algum eventou ou período de matrículas e inscrições, ainda analisa os documento recebidos de maneira manual e sob a análise de um profissional - tomando assim muito tempo. Dito isso, surge a necessidade de um sistema possa resolver este problema, ou melhor, um modelo de Machine-Learn. 

De início, o projeto lerá documentos e terá a função de distinguir, dos documentos recebidos, quais são certidões de nascimento, certidões de casamento e uma terceira classe que será a previsão de documentos não identificados(sejam eles fotos embaçadas, muito tremidas ou escuras) ou de documentos diferentes de uma certidão de nascimento e de casamento.

# ===================================
# ELABORAÇÃO DO PROJETO
# ===================================
Antes da criação do modelo, é necessário realizar a busca de dados e tratar o dataset.
Nesse víes, peguei 30 imagens de cada classe e elaborei um código que criasse mais 5 imagens de cada - com o ângulo e clareza diferentes - para que assim fortificasse o dataset com variabilidade de imagens. O nome desta técnica é Data Augmentation, que consiste em pegar uma imagem que recriá-la de outros ângulos e tons.

Também preferi montar o código de maneira em que o processo seja feito de uma vez com todas as classes. Manualmente também é possível, porém isso geraria mais retrabalho, pois em um modelo que possue inúmeras classes um código que já realiza o processamento de forma automática com todas as classes evita o elevado consumo de tempo, processamento e retrabalho, além de facilitar a escalabilidade e manutenibilidade.

# ===================================
# MEMÓRIA DE CÓDIGO
# ===================================
Antes de argumetar sobre o código, foi necessário criar um ambiente para utilizar das bibliotecas do Python. Sob essa ótica, instalei o a biblioteca *Pilow* que é especializada em manipulação de imagens com o Python. Comando utilizado:

- - - > pip install Pillow


Para a criação do código que cria variações das imagens originais, utilizei as seguintes bibliotecas:
from PIL import Image, ImageEnhance
from pathlib import Path

Image > abre, manipula e salva imagens.
ImageEnhance > permite alterar propriedade visuais, por exemplo, brilho e contraste.
Path > permite trabalhar com caminhos de arquivos e diretórios.

Na lógica do programa, foi criada uma lista que recebessce as três classes que o modelo irá prever: nascimento, casamento, outros

No método main(), a função processar_classe() percorre a lista e executa seu bloco para cada elemento, sendo assim, não sendo necessária a reescrita de um código para cada classe. Também, há variáveis que recebem o caminho de forma automática.




(O LEIA-ME será melhorado. Por enquanto, este rascunho ajudará a manter um norte)
