import numpy as np
import pandas as pd
import sqlite3 as sql

from sklearn import linear_model 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import matplotlib.pyplot as plt ### gráficos
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import VarianceThreshold

import _funciones as fn
from sklearn import tree
from sklearn.tree import export_text 


### conectarse a bd
con=sql.connect("db_arcsce") 
cur=con.cursor()

cur.execute("select name from sqlite_master where type='table'")
cur.fetchall()


##### cargar base de datos ####
df=pd.read_sql(" select * from kpi_arc_ff", con)

#### separar base de detos
X=df.drop(['escenario','Costo','Ventas_perdidas','Arcos_faltantes','Costo_ventas_perdidas','Costo_real_de_la_CS',"Tiempo_tot_esc"],axis=1)
y=df['Ventas_perdidas']

sel=VarianceThreshold()
X2=sel.fit_transform(X)
colum_out=sel.get_feature_names_out()
X2=pd.DataFrame(X2, columns=colum_out)
X2.info(verbose=True)

#### y para clasificacion
y_clas= (y>7000000)
y_clas.astype(int)
y_clas.value_counts()

import plotly.express as px
data=pd.DataFrame([y_clas,y]).T
data.columns=['y_clas','y']
fig = px.box(data, x='y_clas', y="y")
fig.show()

##### ajustar modelos de clasifacion #########

mod=tree.DecisionTreeClassifier(max_depth=10, min_samples_leaf=20)
#mod=RandomForestClassifier(min_samples_leaf=1, class_weight={0:20,1:1})

X_train, X_test, y_train, y_test = train_test_split(X2, y_clas, test_size=0.20)

mod.fit(X_train, y_train)
pred=mod.predict(X_test)
cm=metrics.confusion_matrix(y_test,pred, labels=mod.classes_ )
disp=metrics.ConfusionMatrixDisplay(cm, display_labels=mod.classes_)
print(metrics.classification_report(y_test, pred))
disp.plot()
plt.title("Evaluación")
plt.show()

pred=mod.predict(X_train)
cm=metrics.confusion_matrix(y_train,pred, labels=mod.classes_ )
disp=metrics.ConfusionMatrixDisplay(cm, display_labels=mod.classes_)
disp.plot()
plt.title("Entrenamiento")
plt.show()




X2.columns[32]
X2.columns[88]
X2.columns[49]
X2.columns[38]
X2.columns[57]
X2.columns[53]


#### interaccion 1

X2.columns[88]
X2.columns[31]
X2.columns[32]


####### interacción 2
X2.columns[57]
X2.columns[53]
X2.columns[51]

74/228







plt.figure(figsize=(100,100))
tree.plot_tree(mod,fontsize=10,impurity=False,filled=True)
plt.show()



df['Ventas_perdidas'][df]
df['y_clas']=y_clas

df_analisis=df[['y_clas','Ventas_perdidas',X2.columns[88],X2.columns[31],X2.columns[32]]]

df_analisis.to_excel('intera.xlsx', index=False)


df_analisis=df[['y_clas','Ventas_perdidas',X2.columns[57],X2.columns[51],X2.columns[53]]]

df_analisis.to_excel('intera2.xlsx', index=False)


df_analisis=df[['y_clas','Ventas_perdidas',X2.columns[88],X2.columns[58],X2.columns[32]]]

df_analisis.to_excel('intera3.xlsx', index=False)


def model_imp(y, X2,mod, trans_inv=None):
    
    mod.fit(X2,y)
    df_FI=pd.DataFrame()
    df_FI['arc']=mod.feature_names_in_
    
 
    if isinstance(mod,LinearRegression):
        df_FI['imp']=mod.coef_
    else:
        df_FI['imp']=mod.feature_importances_
        
    
    df_FI.sort_values('imp', ascending =False, inplace=True)
    
    return df_FI, mod


df_FI,mod=model_imp(y_train, X_train, mod)
df_FI.to_excel('df_imp.xlsx', index=False)

df_FI,mod=model_imp(y_clas, X2, mod)