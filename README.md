# Projet de classification de Faux/Vrai Billets

## CONTEXTE
L’un des objectifs de la BEAC (Banque des Etats de l’Afrique Centrale) est de mettre en place des méthodes d’identification des contrefaçons des billets en Francs CFA. Dans le cadre de cette lutte, elle souhaite mettre en place un algorithme qui soit capable de différencier automatiquement les vrais des faux billets.

## OBJECTIFS
Lorsqu’un billet arrive, la BEAC dispose d'une machine qui consigne l’ensemble de ses caractéristiques géométriques. Elle a observé des différences de dimensions entre les vrais et les faux billets. Ces différences sont difficilement notables à l’œil nu, mais une machine devrait sans problème arriver à les différencier.
Ainsi, il faudrait construire un algorithme qui, à partir des caractéristiques géométriques d’un billet, serait capable de déterminer s’il s’agit d’un vrai ou d’un faux billet.

## MISSION
Vous êtes un expert data scientist recruté en CDD pour trouver une solution à ce problème en utilisant l’approche d’apprentissage automatique. Le langage de programmation utilisé est Python.

## PARTIE I : CONNAISSANCE DES DONNÉES ET PRÉTRAITEMENT
### 1. Importation des librairies nécessaires pour le projet
```python
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
```

### 2. Importation des données fournies dans un fichier CSV dans une dataframe pandas
```python
df = pd.read_csv('billets.csv', sep=';')
```

### 3. Affichage des premières et dernières lignes des données
```python
df.head()
df.tail()
```

### 4. Nombre de lignes et de colonnes de la dataframe
```python
nombre_ligne, nombre_colonne = df.shape
print(f'Nombre de lignes : {nombre_ligne}\nNombre de colonnes : {nombre_colonne}')
```

### 5. Informations statistiques des variables et interprétation
```python
df.describe()
```

### 6. Affichage des informations générales sur les données
```python
df.info()
```

### 7. Nombre de valeurs manquantes pour chaque colonne
```python
df.isnull().sum()
```

### 8. Remplacement des valeurs manquantes par régression linéaire
```python
train_df = df.dropna()
test_df = df[df["margin_low"].isnull()]

x_train = train_df.drop("margin_low", axis=1)
y_train = train_df["margin_low"]

lr_model = LinearRegression()
lr_model.fit(x_train, y_train)

y_predict = lr_model.predict(test_df.drop("margin_low", axis=1))
df.loc[df.margin_low.isnull(), "margin_low"] = y_predict
```

## PARTIE II : ANALYSE EXPLORATOIRE DES DONNÉES
### I. Analyse univariée

#### 1. Histogramme du nombre de billets pour chaque type (`is_genuine`)
```python
target = df[['is_genuine']]
palette = ['red', 'green']
plt.figure(figsize=(15,5))
plt.subplot(121)
ax = sns.countplot(x="is_genuine", data=df, palette=palette)
for container in ax.containers:
    ax.bar_label(container)
plt.subplot(122)
df["is_genuine"].value_counts().plot(kind="pie", ylabel='', autopct='%1.0f%%', colors=['green', 'red'])
plt.show()
```

#### 2. Histogrammes de distribution des variables numériques
```python
columns = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']
fig, axes = plt.subplots(2, 3, figsize=(15,7))
for i, col in enumerate(columns):
    plt.subplot(2, 3, i+1)
    sns.histplot(data=df, x=col, bins=15, hue='is_genuine')
plt.show()
```

### II. Analyse bivariée
#### 1. Nuage de points entre les variables numériques
```python
sns.pairplot(df, hue='is_genuine', corner=True)
plt.show()
```

#### 2. Matrice de corrélation
```python
plt.figure(figsize=(10, 8))
sns.clustermap(df.corr(), cmap='coolwarm', annot=True)
plt.title('Cluster Map de Corrélation')
plt.show()
```

## PARTIE III : RÉDUCTION DIMENSIONNELLE ET MODÉLISATION

### 1. Réduction dimensionnelle (ACP)
```python
df_ = df.drop(columns=['is_genuine']).values
df_scaled = StandardScaler().fit_transform(df_)
pca = PCA(n_components=2)
pca.fit(df_scaled)
pcs = pca.components_
print(pcs)
```

### 2. Construction et évaluation du modèle de classification
```python
X = df.drop(columns=['is_genuine'])
y = df['is_genuine']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

## CONCLUSION
Ce projet a permis de démontrer comment une approche d’apprentissage automatique peut être utilisée pour distinguer les vrais des faux billets en analysant leurs caractéristiques géométriques. Des méthodes de prétraitement, d’analyse exploratoire, de réduction dimensionnelle et de modélisation ont été appliquées pour fournir une solution efficace.

