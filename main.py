import DataBase
import IA
from matplotlib import pyplot as plt

data = DataBase.binance_data()
data.show()
index = [i for i in range(len(data.X_train[:,:1]))]
plt.plot(index, data.X_train[:,:1], linewidth=1)
colormap = ['b' for _ in range(len(data.X_train[:,:1]))]
for i in range(len(colormap)):
    if data.y_train.flatten()[i]:
        colormap[i]='g'
    else:
        colormap[i]='r'
plt.scatter(index, data.X_train[:,:1], c=colormap, marker = '+')
plt.show()
res = IA.create_res(IA.Neurone.norm(data.X_train.T), data.y_train.T)

