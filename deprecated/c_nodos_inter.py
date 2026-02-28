import numpy as np
import pandas as pd
import sqlite3 as sql

from sklearn import metrics
import matplotlib.pyplot as plt ### gráficos
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import VarianceThreshold

import _funciones as fn
from sklearn import tree
from sklearn.tree import export_text 



###################Lista de nodos con mayor proporcion de fallos #############
#######################################################################
def prior_nodes ( mod, n_prior=10,):

    n_nod=mod.tree_.node_count ## número de nodos
    nodos=mod.tree_.value.reshape([n_nod,2]) ###  observaciones categoria 0, obse cat 1
    prop_f=nodos[:,1]/(nodos[:,0] +nodos[:,1]) ## calcular proporcion cat 1(los que falllaron)
    nodos_mas_f=(-prop_f).argsort()[0:n_prior] #### definir número de nodos a priorizar
    prop_f[nodos_mas_f]

    nodos_prio= pd.DataFrame()

    nodos_prio['prop_col']=prop_f[nodos_mas_f]
    nodos_prio['ind_nodo']=nodos_mas_f
    nodos_prio['n'] = mod.tree_.n_node_samples[nodos_mas_f]

    #nodos_prio.to_excel('resultados\\nodos_prio.xlsx')

    return nodos_prio

    


#####################Extraer las interacciones ###############



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

    l_ind_var=[] ## crear lista de variables para generar el nodo
   

    deep=len(nodes_path) -1 ### cuantos nodos para llegar al seleccionado

    for i in range(deep,-1,-1):  ## desde el nodo final hasta el raiz
        if mod.tree_.children_right[nodes_path[i-1]] == nodes_path[i]: ### para verificar si la variable si el arco falló
            ind_var=mod.tree_.feature[nodes_path[i-1]] ###guardar el número de la variable(arco)
            l_ind_var.append(ind_var) ### agregar arco a lista de arcos que fallaron
           
    if not l_ind_var: ## si ningun arco falló dejar interacción vacía
        inter=[] 
    else:
        rows=x_train.shape[0] ## número de filas para crear vector de 1
        inter=[1]*rows ## vector de 1 para agregar primera interacción con multiplicación
        name_inter="" ### inicalizar variable con nombre de interacciones
        
        for i in l_ind_var:
            inter=inter*x_train[x_train.columns[i]] ## crea interación multiplicando todos los arcos que fallaron
            name_inter+= x_train.columns[i]+"/" ### crea el nombre agregando los nombres de los arcos

    name_inter= name_inter[:-1]
    return l_ind_var, inter, name_inter



#### esta función aplica la extracción de todas las interacciones de los nodos priorizados
def inter_all_nodes(mod, x_train, n_prior=10 ):

    nodos_prior= prior_nodes(mod, n_prior)

    nodes_list= nodos_prior['ind_nodo'].values


    l_failed=[]
    l_inter=[]
    l_name=[]

    for i in nodes_list:
        
        failed, inter, name=arcs_failed(i, mod, x_train)
        l_failed.append(failed)
        l_inter.append(inter)
        l_name.append(name)
        
        interDF=pd.DataFrame(l_inter).T
        interDF.columns = l_name

        
        
        
    return l_failed, interDF , nodos_prior
        


 ####### evaluación de interacciones
 
 
def dif_prop(var_name, X2):
    
    arcs=X2[var_name]
    nc_aok=np.count_nonzero((y_clas==0) & (arcs==0))
    c_aok=np.count_nonzero((y_clas==1) & (arcs==0))
    aok=np.count_nonzero(arcs==0)
    prop_col_aok = c_aok/aok

    nc_af=np.count_nonzero((y_clas==0) & (arcs==1))
    c_af=np.count_nonzero((y_clas==1) & (arcs==1))
    af=np.count_nonzero(arcs==1)
    prop_col_af = c_af/af

    prop_dif =prop_col_af -prop_col_aok 
     
    return(prop_col_af, prop_col_aok , prop_dif, aok, af)


def todos_arcs(X2):
    
    arc_list=[]
    prop_dif_list=[]
    aok_list=[]
    af_list=[]
    prop_col_af_list=[]
    prop_col_aok_list=[]
    

    
    for var_n in X2.columns:
    
        prop_col_af, prop_col_aok, prop_dif, aok, af = dif_prop(var_n, X2)
        arc_list.append(var_n)
        prop_dif_list.append(prop_dif)
        aok_list.append(aok)
        af_list.append(af)
        prop_col_af_list.append( prop_col_af)
        prop_col_aok_list.append( prop_col_aok)
        
    
    
    dif=pd.DataFrame({'arc':arc_list,'prop_col_af': prop_col_af_list, 'prop_col_aok':prop_col_aok_list,'prop_dif': prop_dif_list, 'n_sce_aok':aok_list,'n_sce_af':af_list})
    dif.sort_values('n_sce_af', ascending=False)
            
    
    return dif
        




#### uso final ####


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


#### filtrar arcos sin probabilidad de fallo
sel=VarianceThreshold()
X2=sel.fit_transform(X)
colum_out=sel.get_feature_names_out()
X2=pd.DataFrame(X2, columns=colum_out)
X2.info(verbose=True)

#### y para clasificacion

y_lim = 7000000  #### param

y_clas= (y>y_lim)
y_clas.astype(int)
y_clas.value_counts()

import plotly.express as px
data=pd.DataFrame([y_clas,y]).T
data.columns=['y_clas','y']
fig = px.box(data, x='y_clas', y="y")
fig.show()

##### ajustar modelos de clasifacion #########



### numero de escenarios que se consdiran significativos
n_min_sce=30 ## param

mod=tree.DecisionTreeClassifier( min_samples_leaf=n_min_sce)
mod.fit(X2, y_clas)

###### evaluar modelo #########################
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


 l_failed, interDF, nodos_prior = inter_all_nodes(mod, X2, 10)
 X_intera=pd.concat([X2,interDF], axis=1)
 comp= todos_arcs(X_intera)
 comp.sort_values('prop_dif', ascending=False)
 
 
 
 

