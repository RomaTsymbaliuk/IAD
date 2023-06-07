import pandas as pd
from sklearn.preprocessing import *
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_csv("ContactLens.csv")
df_encoder = pd.DataFrame(OrdinalEncoder().fit_transform(df), columns = df.columns).astype(np.int32).astype(str)

X = df_encoder[['Age', 'SpectaclePrescrip', 'Astigmatism', 'TearProdRate']].copy()
y = df_encoder['ContactLens']

dt = tree.DecisionTreeClassifier()
dt.fit(X, y)

plt.figure(figsize=(10,8))
tree.plot_tree(dt, feature_names=X.columns, class_names=df['ContactLens'], filled=True)
tr2 = sum([ val_a == val_b for val_a, val_b in zip(dt.predict(X), df['ContactLens'])])
print(f'score = {tr2/df.shape[0]:.1%}')