# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:18:34 2020

@author: utilisateur
"""

from pandas import*
from numpy import*
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
'''
Vous avez à disposition un répertoire nommé Data. 
Il existe dans ce répertoire un ensemble de fichiers(credit_immo.*) 
qui représentent les mêmes données avec des extensions 
différentes (csv, json, xls). En utilisant la ou 
les bibliothèques adéquates, chargez et visualisez ces fichiers
'''

df1=pandas.read_csv('credit_immo.csv')
df2=pandas.read_json('credit_immo.json')
df3=pandas.read_excel('credit_immo.xls')

#result = pandas.concat([df1,df2,df3])

#print(df1,df2,df3)

print(df1)
'''
En utilisant les bibliothèques adéquates, créer une base de données
formée de 6 lignes et 4 colonnes. Les colonnes représentent respectivement 
les variables "taux_de_ventes, croissance_vente, ratio_benefice, ratio_perte".
'''
index = ["0","1","2","3","4","5"]
columns=['taux_de_ventes','croissance_vente','ratio_benefice','ratio_perte']
df = pandas.DataFrame(index=index,columns=columns)
print(df)

'''
En utilisant la fonction dataset.reindex() et dataset.isnull(), 
introduire des données manquantes et récupérer les indices des valeurs
manquantes. Puis remplacez les valeurs manquantes par 0 par exemple. 
Puis supprimez ces valeurs manquantes.
 '''
df=df.fillna(0)
print(df)
'''
Importer les bibliothèques adéquates.
Importer le jeu de données (data-set).
Transformer les valeurs manquantes en moyenne (SimpleImputer)
'''

s = SimpleImputer(missing_values=np.nan,strategy='mean')
#df1.iloc[:,:] = s.fit_transform(df1)
s = s.fit(df1[['Niv_Etude_Bac']])
df1['Niv_Etude_Bac'] = s.transform(df1[['Niv_Etude_Bac']])
s = s.fit(df1[['enfant_a_Charge']])
df1['enfant_a_Charge'] = s.transform(df1[['enfant_a_Charge']])
print(df1)

'''
Encoder les valeurs catégoriques (LabelEncoder)
'''
t = LabelEncoder()
df1['Solvable']=t.fit_transform(df1['Solvable'])
print(df1)

'''
Fractionner le jeu de données pour l’entrainement 
et le test (Training and Test set
'''

a = df1.drop('ID_NOM',axis=1)
x = a['Niv_Etude_Bac'].values.reshape(-1, 1)
y = a['Solvable'].values.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(x_test)
'''
mise à l’échelle des features (StandardScaler).
'''
scaler = StandardScaler()
x_test = scaler.fit_transform(x_test)
print("=====================================")
print(x_test)

'''
Visualisation de données
Chargez le fichier Montant_Temps.csv
'''

df4 = pandas.read_csv('Montant_Temps.csv')

'''
découpez vos données en données d’abscisses et d’ordonnées 
qui représentent respectivement le temps et le montant du capital.
'''
x_Montant_Temps = df4.iloc[:,-1]
y_Montant_Temps = df4.iloc[:,0]

'''
Puis tracez le montant du capital en fonction du temps
(avec la focntion plot()) . Puis sauvegardez vos graphiques
'''

plt.xlabel('Temps')
plt.ylabel('montant du capital')
plt.plot(x_Montant_Temps,y_Montant_Temps)


'''
 Rajoutez du style à vos graphiques
'''


'''
 Visualisation de données sous forme de nuage de points
'''
plt.scatter(x_Montant_Temps,y_Montant_Temps)
plt.show()

'''
Vous avez à disposition le fichier nommé tendance_centrale.csv. 
Chargez ces données 
'''

df5 = pandas.read_csv('tendance_centrale.csv')
print(df5)
'''
calcul de la moyenne
'''
print(df5.mean())
'''
calcul de la mediane
'''
print(df5.median())


df5["Age"].value_counts(normalize=True).plot(kind='bar')
plt.show()