# LeitorDocs

O IF de Almenara, em período de cadastro de algum eventou ou período de matrículas e inscrições, ainda analisa os documento recebidos de maneira manual e sob a análise de um profissional - tomando assim muito tempo. Dito isso, surge a necessidade de um sistema possa resolver este problema, ou melhor, um modelo de Machine-Learn. 

De início, o projeto lerá documentos e terá a função de distinguir, dos documentos recebidos, quais são certidões de nascimento, certidões de casamento e uma terceira classe que será a previsão de documentos não identificados(sejam eles fotos embaçadas, muito tremidas ou escuras) ou de documentos diferentes de uma certidão de nascimento e de casamento.

Antes da criação do modelo, é necessário realizar a busca de dados e tratar o dataset.
Nesse víes, peguei 30 imagens de cada classe e elaborei um código que criasse mais 5 imagens de cada - com o ângulo e clareza diferentes - para que assim fortificasse o dataset com variabilidade de imagens. O nome desta técnica é Data Augmentation, que consiste em pegar uma imagem que recriá-la de outros ângulos e tons.

Também preferi montar o código de maneira em que o processo seja feito de uma vez com todas as classes. Manualmente também é possível, porém isso geraria mais retrabalho, pois em um modelo que possue inúmeras classes um código que já realiza o processamento de forma automática com todas as classes evita o elevado consumo de tempo, processamento e retrabalho.


(O LEIA-ME será melhorado. Por enquanto, este rascunho ajudará a manter um norte)
