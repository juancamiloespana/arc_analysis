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


#mod=RandomForestClassifier(min_samples_leaf=1, class_weight={0:20,1:1})



######## modelo entrenamiento y evaluación ###############3333
mod=tree.DecisionTreeClassifier( min_samples_leaf=1)
X_train, X_test, y_train, y_test = train_test_split(X2, y_clas, test_size=0.20)


mod.fit(X_train, y_train)
pred=mod.predict(X_test)
cm=metrics.confusion_matrix(y_test,pred, labels=mod.classes_ )
disp=metrics.ConfusionMatrixDisplay(cm, display_labels=mod.classes_)
metrics.roc_auc_score(y_test, pred)
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


######## modelo datos completos ###############3333

mod=tree.DecisionTreeClassifier( min_samples_leaf=30)
mod.fit(X2, y_clas)
pred=mod.predict(X2)
cm=metrics.confusion_matrix(y_clas,pred, labels=mod.classes_ )
disp=metrics.ConfusionMatrixDisplay(cm, display_labels=mod.classes_)
metrics.roc_auc_score(y_clas, pred)
print(metrics.classification_report(y_clas, pred))
disp.plot()
plt.title("Evaluación")
plt.show()


plt.figure(figsize=(100,100))
tree.plot_tree(mod,fontsize=10,impurity=False,filled=True,node_ids=True)
plt.show()



###################Lista de nodos con mayor proporcion de fallos #############
#######################################################################

n_prior=10 ### nodos a seleccionar
n_nod=mod.tree_.node_count ## número de nodos
nodos=mod.tree_.value.reshape([n_nod,2]) ###  observaciones categoria 0, obse cat 1
prop_f=nodos[:,1]/(nodos[:,0] +nodos[:,1]) ## calcular proporcion cat 1(los que falllaron)
nodos_mas_f=(-prop_f).argsort()[0:n_prior] #### definir número de nodos a priorizar
prop_f[nodos_mas_f]

nodos_prio= pd.DataFrame()

nodos_prio['prop_col']=prop_f[nodos_mas_f]
nodos_prio['ind_nodo']=nodos_mas_f
nodos_prio['n'] = mod.tree_.n_node_samples[nodos_mas_f]

nodos_prio.to_excel('resultados\\nodos_prio.xlsx')


#####################Extraer las interacciones ###############

nodo=nodos_mas_f[0]  ### seleccionar el número de un nodo para conocer la ruta o interacciones

nodo=234


def arcs_failed(nodo, mod, x_train):
    dp=mod.decision_path(x_train) ### decision path 

    dp.indices.shape ##el camino para llegar al nodo final de esa observacion, se repite la observacion por cada nodo en el que está
    dp.indptr.shape ### el indice en el que arrancha el camino de cada observación 



    indice_fin=np.where(dp.indices==nodo)[0][0] ### en qué indice de nodos está el nodo seleccionado
    dp.indices[1760:indice_fin] ## para comprobar que el numero del nodo según el indice coincide con el seleccionado

    dif=[x for x in indice_fin-dp.indptr if x>0]

    pos_ini=np.argmin(dif) ### indica la posicion de indptr en la que está el indice inicial (el cero más cercano para determinar el path)
    indice_ini=dp.indptr[pos_ini] ## extraer el indice en el que arranca el path

    nodes_path=dp.indices[indice_ini:(indice_fin+1)] ## crea una lista con el camino desde 0 al nodo indicado

    l_ind_var=[] ## crear lista de variables en 1 
    deep=len(nodes_path) -1
    i=1
    for i in range(deep,-1,-1):
        if mod.tree_.children_right[nodes_path[i-1]] == nodes_path[i]:
            ind_var=mod.tree_.feature[nodes_path[i-1]]
            l_ind_var.append(ind_var)

    return l_ind_var



failed=arcs_failed(233, mod, X2)

if not failed:
    inter=[]
    
else:
    rows=X2.shape[0]
    inter=[1]*rows
    for i in failed:
        inter=inter*X2[X2.columns[i]]

np.unique(inter, return_counts=True)



#########


inter1=X2[X2.columns[49]]* X2[X2.columns[31]]* X2[X2.columns[88]]
val= inter1==inter

np.unique(val, return_counts=True)

################################################3

###### crear interacciones ###################333

df_analisis=df[['y_clas','Ventas_perdidas',X2.columns[88],X2.columns[31],X2.columns[32]]]

df_analisis.to_excel('intera.xlsx', index=False)


df_analisis=df[['y_clas','Ventas_perdidas',X2.columns[57],X2.columns[51],X2.columns[53]]]

df_analisis.to_excel('intera2.xlsx', index=False)


df_analisis=df[['y_clas','Ventas_perdidas',X2.columns[88],X2.columns[58],X2.columns[32]]]

df_analisis.to_excel('intera3.xlsx', index=False)



############################################################
############### Medir importancia #########################
##########################################################

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