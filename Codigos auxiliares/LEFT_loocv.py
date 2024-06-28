from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv1D, Dense, Reshape, Flatten
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import LeaveOneOut

def normalizar_sinal(sinal):
    valor_minimo = np.min(sinal)
    valor_maximo = np.max(sinal)
    sinal_normalizado = (sinal - valor_minimo)/(valor_maximo - valor_minimo)

    return sinal_normalizado

def padronizar_sinal(sinal):
    media = np.mean(sinal)
    desvio_padrao = np.std(sinal)
    sinal_padronizado = (sinal - media)/desvio_padrao

    return sinal_padronizado

def separar_sinal(sinal):

    sinal_separado = []
    for i in range(int(len(sinal)/3)):

        if len(sinal) < (i + 1)*90:
            break
        
        inicio = i * 90
        fim = (i + 1) * 90
        
        sinal_separado.append(sinal[inicio : fim])

    sinal_separado = np.array(sinal_separado)

    return sinal_separado

def preprocessar_sinal_RGB(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo)

    sinal_R = df['R']
    sinal_G = df['G']
    sinal_B = df['B']

    sinal_normalizado_R = normalizar_sinal(sinal_R)
    sinal_normalizado_G = normalizar_sinal(sinal_G)
    sinal_normalizado_B = normalizar_sinal(sinal_B)

    sinal_padronizado_R = padronizar_sinal(sinal_normalizado_R)
    sinal_padronizado_G = padronizar_sinal(sinal_normalizado_G)
    sinal_padronizado_B = padronizar_sinal(sinal_normalizado_B)

    sinal_separado_R = separar_sinal(sinal_padronizado_R)
    sinal_separado_G = separar_sinal(sinal_padronizado_G)
    sinal_separado_B = separar_sinal(sinal_padronizado_B)

    dim0, dim1 = sinal_separado_B.shape
    entrada_rede = np.zeros((dim0, dim1, 3))

    entrada_rede[:, :, 0] = sinal_separado_R
    entrada_rede[:, :, 1] = sinal_separado_G
    entrada_rede[:, :, 2] = sinal_separado_B

    return entrada_rede

def criar_modelo():
    modelo = Sequential([
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.1), input_shape=(90, 3, 1)),
    Reshape((90 - 2, 64)),
    Conv1D(64, 12, activation='relu', kernel_regularizer=l2(0.1)),
    Conv1D(32, 12, activation='relu', kernel_regularizer=l2(0.1)),
    Flatten(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.1)),
    Dense(1, activation=None) 
])
    
    return modelo

def lr_decay(epoca):
    if epoca < 80:
        return 0.00001
    else:
        return 0.00001 * tf.math.exp(-0.1)

def saturacoes(caminho_csv, sinal):
    df_saturacao = pd.read_csv(caminho_csv)
    saturacoes = (df_saturacao['SpO2 1'] + df_saturacao['SpO2 2'] + df_saturacao['SpO2 4'] + df_saturacao['SpO2 5'])/4

    saturacoes_3s = []

    for i in range(0, len(saturacoes), 3):
        saturacao_3s = np.mean(saturacoes[i:i + 4])

        if i + 3 < len(saturacoes):
            saturacoes_3s.append(saturacao_3s)

    saturacoes_3s = np.array(saturacoes_3s)
    dim0_sat = len(saturacoes_3s)
    dim0_sinal, dim1_sinal, dim3_sinal = sinal.shape

    if dim0_sat > dim0_sinal:
        saturacoes_3s = saturacoes_3s[:-1]

    return saturacoes_3s


caminho_100001 = r"D:\usuarios\pedro.watanabe\Meus Documentos\Projetos\RedeNeural\Experiments\Atividade 22\ppg_csv\Left\100001.csv"
caminho_100002 = r"D:\usuarios\pedro.watanabe\Meus Documentos\Projetos\RedeNeural\Experiments\Atividade 22\ppg_csv\Left\100002.csv"
caminho_100003 = r"D:\usuarios\pedro.watanabe\Meus Documentos\Projetos\RedeNeural\Experiments\Atividade 22\ppg_csv\Left\100003.csv"
caminho_100004 = r"D:\usuarios\pedro.watanabe\Meus Documentos\Projetos\RedeNeural\Experiments\Atividade 22\ppg_csv\Left\100004.csv"
caminho_100005 = r"D:\usuarios\pedro.watanabe\Meus Documentos\Projetos\RedeNeural\Experiments\Atividade 22\ppg_csv\Left\100005.csv"
caminho_100006 = r"D:\usuarios\pedro.watanabe\Meus Documentos\Projetos\RedeNeural\Experiments\Atividade 22\ppg_csv\Left\100006.csv"


caminho_saturacao_100001 = r"D:\usuarios\pedro.watanabe\Meus Documentos\Projetos\RedeNeural\Experiments\Atividade 22\gt\100001.csv"
caminho_saturacao_100002 = r"D:\usuarios\pedro.watanabe\Meus Documentos\Projetos\RedeNeural\Experiments\Atividade 22\gt\100002.csv"
caminho_saturacao_100003 = r"D:\usuarios\pedro.watanabe\Meus Documentos\Projetos\RedeNeural\Experiments\Atividade 22\gt\100003.csv"
caminho_saturacao_100004 = r"D:\usuarios\pedro.watanabe\Meus Documentos\Projetos\RedeNeural\Experiments\Atividade 22\gt\100004.csv"
caminho_saturacao_100005 = r"D:\usuarios\pedro.watanabe\Meus Documentos\Projetos\RedeNeural\Experiments\Atividade 22\gt\100005.csv"
caminho_saturacao_100006 = r"D:\usuarios\pedro.watanabe\Meus Documentos\Projetos\RedeNeural\Experiments\Atividade 22\gt\100006.csv"


X_1 = preprocessar_sinal_RGB(caminho_100001)
X_2 = preprocessar_sinal_RGB(caminho_100002)
X_3 = preprocessar_sinal_RGB(caminho_100003)
X_4 = preprocessar_sinal_RGB(caminho_100004)[:-1]
X_5 = preprocessar_sinal_RGB(caminho_100005)
X_6 = preprocessar_sinal_RGB(caminho_100006)

y_1 = saturacoes(caminho_saturacao_100001, X_1)
y_2 = saturacoes(caminho_saturacao_100002, X_2)
y_3 = saturacoes(caminho_saturacao_100003, X_3)
y_4 = saturacoes(caminho_saturacao_100004, X_4)
y_5 = saturacoes(caminho_saturacao_100005, X_5)
y_6 = saturacoes(caminho_saturacao_100006, X_6)


indices = [1, 2, 3, 4, 5, 6]
maes = []
for indice in indices:

    if indice == 1:
        X_train = np.concatenate((X_2, X_3, X_4, X_5, X_6), axis=0)
        y_train = np.concatenate((y_2, y_3, y_4, y_5, y_6), axis=0)

        X_test = X_1
        y_test = y_1

    if indice == 2:
        X_train = np.concatenate((X_1, X_3, X_4, X_5, X_6), axis=0)
        y_train = np.concatenate((y_1, y_3, y_4, y_5, y_6), axis=0)

        X_test = X_2
        y_test = y_2

    if indice == 3:
        X_train = np.concatenate((X_2, X_1, X_4, X_5, X_6), axis=0)
        y_train = np.concatenate((y_2, y_1, y_4, y_5, y_6), axis=0)

        X_test = X_3
        y_test = y_3

    if indice == 4:
        X_train = np.concatenate((X_2, X_3, X_1, X_5, X_6), axis=0)
        y_train = np.concatenate((y_2, y_3, y_1, y_5, y_6), axis=0)

        X_test = X_4
        y_test = y_4

    if indice == 5:
        X_train = np.concatenate((X_2, X_3, X_4, X_1, X_6), axis=0)
        y_train = np.concatenate((y_2, y_3, y_4, y_1, y_6), axis=0)

        X_test = X_5
        y_test = y_5

    if indice == 6:
        X_train = np.concatenate((X_2, X_3, X_4, X_1, X_5), axis=0)
        y_train = np.concatenate((y_2, y_3, y_4, y_1, y_5), axis=0)

        X_test = X_6
        y_test = y_6

    modelo = criar_modelo()
    lr_callback = LearningRateScheduler(lr_decay)

    modelo.compile(optimizer='adam', loss='mse')

    modelo.fit(X_train, y_train, epochs=200, callbacks=[lr_callback])

    y_pred = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    maes.append(mae)

print(maes)
print("Media mae: ", np.mean(maes))
print("Desvio Padrao: ", np.std(maes))

