

from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import sqlite3 as sql

from sklearn import linear_model 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt ### gráficos


from sklearn.ensemble import RandomForestRegressor

from sklearn.feature_selection import VarianceThreshold

import _funciones as fn
from sklearn import tree
from sklearn.tree import export_text 







##################################################################
##################################################################
##################Conectarse BD y revisar tablas##################
##################################################################

### conectarse a bd
con=sql.connect("db_arcsce") 
cur=con.cursor()

cur.execute("select name from sqlite_master where type='table'")
cur.fetchall()


###### tabla ####

df=pd.read_sql(" select * from kpi_arc_ff", con)


#### separar base de detos

X=df.drop(['escenario','Costo','Ventas_perdidas','Arcos_faltantes','Costo_ventas_perdidas','Costo_real_de_la_CS',"Tiempo_tot_esc"],axis=1)
y=df['Ventas_perdidas']


############

X.info(verbose=True)
sel=VarianceThreshold()
X2=sel.fit_transform(X)
colum_out=sel.get_feature_names_out()
X2=pd.DataFrame(X2, columns=colum_out)
X2.info(verbose=True)
sel.get_feature_names_out()
arc_sel=pd.DataFrame()
arc_sel['arc']=sel.feature_names_in_
arc_sel['conserva']=sel.get_support()
arc_sel.sort_values(by='conserva')

##################################################################
##################################################################
##################Ajustar modelos ###################### #########
##################################################################


####### ajustar modelos sin transformar variable respuesta #################3


def eval(model,y,X2, trans_inv=None):
    
    y_pred=model.predict(X2)
        
    if trans_inv is not None:
        
        
        y_pred_i=np.exp(y_pred)
        y_i=np.exp(y)
        ind={'mape': mean_absolute_percentage_error(y_i,y_pred_i),
            'mae':mean_absolute_error(y_i,y_pred_i),
            'rmse': np.sqrt(mean_squared_error(y_i,y_pred_i)),
            'mape_ylog': mean_absolute_percentage_error(y,y_pred),
            'mae_y_log':mean_absolute_error(y,y_pred),
            'rmse_y_log': np.sqrt(mean_squared_error(y,y_pred))
            
            }
        
    else:
        
        ind={'mape': mean_absolute_percentage_error(y,y_pred),
            'mae':mean_absolute_error(y,y_pred),
            'rmse': np.sqrt(mean_squared_error(y,y_pred))
            }
        
    return ind


def rl_imp(y, X2, trans_inv=None):
    reg=LinearRegression(fit_intercept= False)
    reg.fit(X2,y)
    coef=reg.coef_
    name=reg.feature_names_in_

    df_coef=pd.DataFrame()
    df_coef['arc']=name
    df_coef['coef']=coef

    df_coef.sort_values(by='coef', ascending=False, inplace=True)

    ind =eval(reg, y, X2,trans_inv)
    
    return df_coef, reg, ind


def rf_imp(y,X2, trans_inv=None):
    
    RF=RandomForestRegressor(n_estimators=100)
    RF.fit(X2,y)
 
    df_FI_RF=pd.DataFrame()
    df_FI_RF['arc']=RF.feature_names_in_
    df_FI_RF['coef']=RF.feature_importances_

    df_FI_RF.sort_values('coef',ascending=False, inplace =True)
    ind =eval(RF, y, X2,trans_inv)
    return df_FI_RF, RF, ind



def model_imp(y, X2,mod, trans_inv=None):
    
    mod.fit(X2,y)
    df_FI=pd.DataFrame()
    df_FI['arc']=mod.feature_names_in_
    
 
    if isinstance(mod,LinearRegression):
        df_FI['imp']=mod.coef_
    else:
        df_FI['imp']=mod.feature_importances_
        
    
    df_FI.sort_values('imp', ascending =False, inplace=True)
    ind =eval(mod, y, X2,trans_inv)
    
    return df_FI, mod, ind




import openpyxl

#####regresion lineal
regdf_coef, _, ind_y=rl_imp(y,X2)
regdf_coef.to_excel('reg_orig.xlsx', index=False)
y_log=np.log(y+5)
regdf_coef, _, ind_y=rl_imp(y_log,X2, trans_inv=True)
regdf_coef.to_excel('reg_orig.xlsx', index=False)

#######Random forest ###############

df,_,ind_y=rf_imp(y,X2)
df.to_excel('df_imp.xlsx', index=False)


df,_,ind_y=rf_imp(y_log,X2, trans_inv=True)
df.to_excel('df_imp.xlsx', index=False)

np.mean(y)
np.std(y)
np.median(y)

########## validar importancia en arboles de decisions
rt=tree.DecisionTreeRegressor(min_samples_leaf=1)
df,_,ind_y=model_imp(y,X2, rt)
df.to_excel('df_imp.xlsx', index=False)











##########PCA ##### PCA no funcionó, las variablrs tienen informacion muy diferente
X2
X2.columns
pca = PCA(n_components = 70)


X_pca = pca.fit_transform(X2)
xplained_variance = pca.explained_variance_ratio_
xplained_variance.sum()


##########################################
X_pca_DF=pd.DataFrame(X_pca, columns=pca_n)

reg_y=rl_imp(y,X_pca_DF)
RF_y=rf_imp(y,X2)

reg_logy=rl_imp(y_log,X2, trans_inv=True)
RF_logy=rf_imp(y_log,X2, trans_inv=True)



####### probar otros modelos ######

ylog=np.log(y)
clf = tree.DecisionTreeRegressor(min_samples_leaf= 100)
clf.get_params()
clf = clf.fit(X2, y)
y_pred=clf.predict(X2)
clf.get_depth()
np.sqrt(mean_squared_error(y,y_pred))

pred_df=pd.DataFrame({'y_pred':y_pred, 'y':np.array(y)})

pred_df['error']=pred_df['y']-pred_df['y_pred']

qmax=pred_df['error'].quantile(0.95)
qmin=pred_df['error'].quantile(0.05)

error=pred_df[(pred_df['error']<=qmax) & (pred_df['error']>=qmin )]
np.sqrt(mean_squared_error(error['y'],error['y_pred']))

y.quantile(0.9)
y.hist(bins=100)
y.to_excel('ventas_perdidas.xlsx')

y_clas= (y<7000000)
y_clas.astype(int)
y_clas.value_counts()

unicos=pd.DataFrame(np.round(y.unique()/10000,0)*10000)
unicos.to_excel('ventas_perdidas_unica.xlsx')

error['error'].hist()

error['error'].min()

clf.feature_importances_
clf.feature_names_in_

feat_imp=pd.DataFrame({'var':clf.feature_names_in_,'imp':clf.feature_importances_})

feat_imp.sort_values(by='imp',ascending=False)

r = export_text(clf,feature_names=X2.columns.tolist(),show_weights=True)
print(r)
plt.figure(figsize=(40,40))
tree.plot_tree(clf,fontsize=9,impurity=False,filled=True)
plt.show()

X2.columns[32]
X2.columns[88]
X2.columns[49]
X2.columns[38]
X2.columns[57]
X2.columns[53]

X2[X2.columns[88]].sum()

box_df=df[['Ventas_perdidas', 'Caloto - Popayan']]
box_df['y_log']=np.log(box_df['Ventas_perdidas'])
box_df2=box_df[box_df['Ventas_perdidas']<=10000000]


box_df[['y_log','Caloto - Popayan']].boxplot(by='Caloto - Popayan')

box_df2[['Ventas_perdidas','Caloto - Popayan']].boxplot(by='Caloto - Popayan')


inter_df=df[['Ventas_perdidas',X2.columns[88], X2.columns[32]]]

inter_df['sum']=inter_df[X2.columns[88]]+inter_df[X2.columns[32]]

inter_df2=inter_df[inter_df['Ventas_perdidas']<=10000000][['Ventas_perdidas','sum']]
inter_df2=inter_df[['Ventas_perdidas','sum']]

inter_df2.boxplot(by='sum')
X2.columns[49]


####### redes neuronales ##

import tensorflow as ts
from tensorflow import keras 
from tensorflow.keras import layers







