

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer ### para imputación
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
import joblib
from sklearn import linear_model 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

####Este archivo contienen funciones utiles a utilizar en diferentes momentos del proyecto

###########Esta función permite ejecutar un archivo  con extensión .sql que contenga varias consultas



def ejecutar_sql (nombre_archivo, cur):
    sql_file=open(nombre_archivo)
    sql_as_string=sql_file.read()
    sql_file.close
    cur.executescript(sql_as_string)
    
  
  
def imputar_f (df,list_cat):  
        
    
    df_c=df[list_cat]
    df_n=df.loc[:,~df.columns.isin(list_cat)]

    imputer_n=SimpleImputer(strategy='median')
    imputer_c=SimpleImputer(strategy='most_frequent')

    imputer_n.fit(df_n)
    imputer_c.fit(df_c)

    X_n=imputer_n.transform(df_n)
    X_c=imputer_c.transform(df_c)

    df_n=pd.DataFrame(X_n,columns=df_n.columns)
    df_c=pd.DataFrame(X_c,columns=df_c.columns)

    df =pd.concat([df_n,df_c],axis=1)
    return df


def sel_variables(modelos,X,y,threshold):
    
    var_names_ac=np.array([])
    for modelo in modelos:
        #modelo=modelos[i]
        modelo.fit(X,y)
        sel = SelectFromModel(modelo, prefit=True,threshold=threshold)
        var_names= sel.get_feature_names_out(modelo.feature_names_in_)
        var_names_ac=np.append(var_names_ac, var_names)
        var_names_ac=np.unique(var_names_ac)
    
    return var_names_ac


def medir_modelos(modelos,scoring,X,y,cv):

    metric_modelos=pd.DataFrame()
    for modelo in modelos:
        scores=cross_val_score(modelo,X,y, scoring=scoring, cv=cv )
        pdscores=pd.DataFrame(scores)
        metric_modelos=pd.concat([metric_modelos,pdscores],axis=1)
    
    metric_modelos.columns=["reg_lineal","decision_tree","random_forest","gradient_boosting"]
    return metric_modelos



def preparar_datos (df):
   
    

    #######Cargar y procesar nuevos datos ######
   
    
    #### Cargar modelo y listas 
    
   
    list_cat=joblib.load("list_cat.pkl")
    list_dummies=joblib.load("list_dummies.pkl")
    var_names=joblib.load("var_names.pkl")

    ####Ejecutar funciones de transformaciones
    
    df=funciones.imputar_f(df,list_cat,SimpleImputer,pd)
    df_dummies=pd.get_dummies(df,columns=list_dummies)
    df_dummies=df_dummies[var_names]
    
    return df_dummies


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

def model_imp_nind(y, X2,mod, trans_inv=None):
    
    mod.fit(X2,y)
    df_FI=pd.DataFrame()
    df_FI['arc']=mod.feature_names_in_
    
 
    if isinstance(mod,LinearRegression):
        df_FI['imp']=mod.coef_
    else:
        df_FI['imp']=mod.feature_importances_
        
    
    df_FI.sort_values('imp', ascending =False, inplace=True)
 
    
    return df_FI, mod