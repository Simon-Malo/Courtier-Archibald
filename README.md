# Courtier-Archibald

## Data

Il est posible d'utiliser des données sur les cours des différentes cryptos en utilisant la classe data définie dans le main.
```
data = DataBase.binance_data()
```
Cette ligne va importer un DataFrame pandas contenant les informations espacées par un intervalle d'une heure sur le bitcoin avec quelques indicateurs calculés. Les plus importants sont dans la variable statique "frame" et les données sont toutes répertoriées dans le DataFrame "df" : 
```
tableau_des_valeurs_les_plus_importantes = data.df #Tableau contenant : [Open, Volume, RSI, diff_M_G(différence entre moyenne glissante et open), Vortex (différence entre les deux courbes de vortex)]
Toutes_les_valeurs = data.frame [Mêmes valeurs que dans frame + Close, High, Low]
```
Des méthodes sont utilisables, voir le nom des fonctions pour comprendre. 
