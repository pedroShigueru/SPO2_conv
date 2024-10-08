**LOOCV_right.py** é o arquivo que contém o código principal

Artigo utilizado para o estudo e aprendizado:
Hoffman, J.S., Viswanath, V.K., Tian, C. et al. Smartphone camera oximetry in an induced hypoxemia study. npj Digit. Med. 5, 146 (2022). https://doi.org/10.1038/s41746-022-00665-y

Tarefas realizadas: 
<br/>
<br/>
**1) Pré processamento dos sinais de ppg**<br/>
    - Normalização <br/>
    - Padronização <br/>
    - Janelamento de 3 segundos do sinal 
    <br/>
    <br/>
**2) Modelagem**<br/>
    - Houve uma mistura de Convolução 2D e 1D<br/>
    - Camadas Densas para regressão<br/>
    - Leave One Out Cross Validation (LOOCV)
<br/>
<br/>
Usando esse método, foi obtido um resultado bastante satisfatório. A média da métrica Mean Absolute Error foi de 4.25 

Dataset -> "raw", "ppg_csv", "gt"

raw -> contém vídeos do dedo dos pacientes, gravados utilizando um celular com flash.

ppg_csv -> contém sinais de ppg. Cada amostra do sinal foi obtidos através da média de um frame.

gt -> contém os valores de saturação de oxigênio captados utilizando um oxímetro, durante a gravação do vídeo.
