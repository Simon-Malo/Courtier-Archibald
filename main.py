import DataBase
import IA

data = DataBase.binance_data()
IA.create_res(IA.Neurone.norm(data.X_train.T), data.y_train)

