import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(r"D:\usuarios\pedro.watanabe\Meus Documentos\Projetos\RedeNeural\Experiments\Atividade 22\ppg_csv\Left\100001.csv")

red = df['R'][0:1000]
green = df['G'][0:1000]
blue = df['B'][0:1000]

fig, axs = plt.subplots(3)
axs[0].plot(red)
axs[1].plot(green)
axs[2].plot(blue)
plt.show()