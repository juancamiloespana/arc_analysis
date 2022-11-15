

import numpy as np
import pandas as pd
import sqlite3 as sql

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.feature_selection import VarianceThreshold



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

### eliminar columnas constantes

X=df.drop(['escenario','Costo','Ventas_perdidas','Arcos_faltantes','Costo_ventas_perdidas','Costo_real_de_la_CS',"Tiempo_tot_esc"],axis=1)
y=df['Ventas_perdidas']


######
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
            'mape_ylog': mean_absolute_percentage_error(y,y_pred),
            'mae_y_log':mean_absolute_error(y,y_pred)}
        
    else:
        
        ind={'mape': mean_absolute_percentage_error(y,y_pred),
            'mae':mean_absolute_error(y,y_pred)}
        
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


reg_y=rl_imp(y,X2)
RF_y=rf_imp(y,X2)

reg_logy=rl_imp(y_log,X2, trans_inv=True)
RF_logy=rf_imp(y_log,X2, trans_inv=True)





