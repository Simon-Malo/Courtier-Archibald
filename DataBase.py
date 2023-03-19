from binance.client import Client
import pandas as pd
import json
client = Client('DJbwB7qN4CaGMb32UYdHjfGF1TNEuD2nUZZaFEiVLzuvjymI7yC9vVdXRJi8C94l','i9EREf453jKRciBZuidEmvN6e2xpSkYuOusifKlDT1OKeLh58eTie0GbrL9fdyeP')
account = client.get_account()
from matplotlib import pyplot as plt
import numpy as np
import os

class binance_data:
    def __init__(self):
        self.path = os.getcwd()+"/BITCOIN.json"
        try :
            with open(self.path, "r") as file:
                jsonStr = file.read()
                self.df = pd.read_json(jsonStr)
        except:
            self.df = self.get_minute_data("BTCUSDT", "10 years ago UTC")
            #Calcul du RSI
            rsi = self.RSI(self.df)
            #Ajout de la colonne au df
            self.df["RSI"] = rsi
            #traitement des données (ajustement des lignes en fonction de l'Open)
            rsi = self.df.RSI.tolist()
            while rsi[-1] == "NaN":
                rsi.insert(0, rsi.pop())
            self.df['RSI'] = rsi
            #Calcul de la moyenne glissante
            valeurs = self.df.Open.tolist()
            intervalle = 3
            mg = self.moyenne_glissante(valeurs, intervalle)
            #Traitement des données (ajustement des lignes en fonction de l'Open)
            for i in range(len(self.df.index) - len(mg)):
                mg.insert(0, "NaN")
            diff_m_g = []
            for i in range(len(mg)):
                if mg[i] != "NaN":
                    diff_m_g.append(valeurs[i] - mg[i])
                else:
                    diff_m_g.append("NaN")
            self.df['diff_M_G'] = diff_m_g
            vortex = self.Vortex(self.df, 21)
            vrtx = []
            for i in range(len(vortex[0])):
                if vortex[0][i] != "NaN":
                    vrtx.append(float(vortex[0][i]) - float(vortex[1][i]))
            self.df['Vortex'] = vrtx
            #Enregistrement du fichier sous forme de JSON
            self.df.to_json(self.path)
        self.frame = self.df[['Open', 'Volume', "RSI", "diff_M_G", "Vortex"]]
        print(self.frame)

    def get_infos(self,symbol,interval,lookback,datetime=True):
      frame = pd.DataFrame(client.get_historical_klines(symbol,interval,lookback))
      frame = frame.iloc[:,:6]
      frame.columns = ['Time','Open','High','Low','Close','Volume']
      frame = frame.set_index('Time')
      if datetime:
        frame.index = pd.to_datetime(frame.index, unit = 'ms')
      frame = frame.astype(float)
      return frame

    def get_minute_data(self,symbol,lookback,datetime=True):
      info = client.get_symbol_info(symbol)
      frame = self.get_infos(symbol, '1h', lookback,datetime=datetime)
      frame.Open.plot()
      return frame

    def RSI(self, df):
        change = df["Close"].diff()
        change.dropna(inplace=True)
        # Create two copies of the Closing price Series
        change_up = change.copy()
        change_down = change.copy()

        change_up[change_up<0] = 0
        change_down[change_down>0] = 0

        # Verify that we did not make any mistakes
        change.equals(change_up+change_down)

        # Calculate the rolling average of average up and average down
        avg_up = change_up.rolling(14).mean()
        avg_down = change_down.rolling(14).mean().abs()

        rsi = 100 * avg_up / (avg_up + avg_down)
        return rsi

    def moyenne_glissante(self, valeurs, intervalle):
        indice_debut = (intervalle - 1) // 2
        liste_moyennes = [sum(valeurs[i - indice_debut:i + indice_debut + 1]) / intervalle for i in
                          range(indice_debut, len(valeurs) - indice_debut)]
        return liste_moyennes

    def Vortex(self, frame,  periods):
        vectp = ["Nan" for i in range(periods)]
        vectm = ["Nan" for i in range(periods)]
        high = frame.High.tolist()
        close = frame.Close.tolist()
        low = frame.Low.tolist()
        TR = [] ; VMp = [] ; VMm = []
        for i in range(1,len(high)):
            TR.append(max(abs(high[i] - close[i-1]), abs(high[i] - low[i]), abs(low[i] - close[i-1])))
            VMp.append(abs(high[i] - low[i-1])) ; VMm.append(abs(low[i] - high[i-1]))
            if i >= periods:
                TR_P = np.sum(TR[i-periods:i])
                VMp_P = np.sum(VMp[i - periods:i])
                VMm_P = np.sum(VMm[i - periods:i])
                vectp.append(VMp_P/TR_P)
                vectm.append(VMm_P / TR_P)
        return (vectp, vectm)




def plot_rsi(frame, rsi):
    # Set the theme of our chart
    plt.style.use('fivethirtyeight')

    # Make our resulting figure much bigger
    plt.rcParams['figure.figsize'] = (20, 20)
    # Create two charts on the same figure.
    ax1 = plt.subplot2grid((10,1), (0,0), rowspan = 4, colspan = 1)
    ax2 = plt.subplot2grid((10,1), (5,0), rowspan = 4, colspan = 1)

    # First chart:
    # Plot the closing price on the first chart
    ax1.plot(frame['Close'], linewidth=2)
    ax1.set_title('Bitcoin Close Price')

    # Second chart
    # Plot the RSI
    ax2.set_title('Relative Strength Index')
    ax2.plot(rsi, color='orange', linewidth=1)
    # Add two horizontal lines, signalling the buy and sell ranges.
    # Oversold
    ax2.axhline(30, linestyle='--', linewidth=1.5, color='green')
    # Overbought
    ax2.axhline(70, linestyle='--', linewidth=1.5, color='red')
    # Display the charts
    plt.show()

