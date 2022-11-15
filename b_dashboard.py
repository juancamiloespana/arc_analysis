
##################################################################
##################################################################
##################Cargar paquetes ################################
##################################################################

### paquetes con operaciones básicas y sql 
import pandas as pd
import sqlite3 as sql ##para conectarse a bd, traer y manipular info con sql
import numpy as np
import math ### para floor y ceil

#### para hacer gráficas

import plotly.express as px



from statistics import linear_regression
from tabnanny import verbose
import streamlit as st


import plotly.graph_objects as go
import matplotlib





##################################################################
##################################################################
##################Conectarse BD y revisar tablas##################
##################################################################

### conectarse a bd
con=sql.connect("db_arcsce") 
cur=con.cursor()

cur.execute("select name from sqlite_master where type='table'")
cur.fetchall()


####### traer trablas de arcos 

info_arc=pd.read_sql("select * from info_arc", con).sort_values(by='prob_fallo')
info_nodes=pd.read_sql("select * from info_nodes", con)


##################################################################
##################################################################
##################preprocesar y limpirar tablas de datos #########
##################################################################

##### traer tabla de kpi y cambiar formatos de campos #####
#pd.set_option('display.max_rows', None) ### para que muestre todos los registros hace muy lento la ejecución



###### traer tabla de arcos y escenarios y cruzar con KPI

df=pd.read_sql(" select * from kpi_arc_ff", con)


#######Explorar variable respuesta ########

####crear base arcos y variable respuesta######
max=math.floor(df['Ventas_perdidas'].max())
df['Ventas_perdidas'].value_counts().sort_index()

df[df['Ventas_perdidas']==0].shape ### 626 escenario sin ventas perdidas
df[df['Ventas_perdidas']>=max].shape### 591 escenarios con colapso de demanda


X=df.drop(['escenario','Costo','Ventas_perdidas','Arcos_faltantes','Costo_ventas_perdidas','Costo_real_de_la_CS',"Tiempo_tot_esc"],axis=1)
y=df['Ventas_perdidas']


prob_fallo_sim=X.sum()/10083
prob_fallo_sim.sort_values(ascending=False)

prob_fallo=prob_fallo_sim.to_frame(name="prob_fall_sim").reset_index()
prob_fallo.rename(columns={'index':'arc'}, inplace=True)

prob_fallos=prob_fallo.merge(info_arc[['arc','prob_fallo']], how='inner', on= 'arc' ).sort_values('prob_fallo', ascending=False)
prob_fallos['prop_fallos']=prob_fallos['prob_fall_sim']/prob_fallos['prob_fallo']
prob_fallos

##################################################################
##################################################################
##################Explorar Datos################################
##################################################################

###### explorar ventas perdidas ########

fig=px.histogram(df, x='Ventas_perdidas',nbins=30)
fig.update_traces(marker=dict(line=dict(width=0.5,color='black')),
                  selector=dict(type='histogram'))
fig.show()


##### correlaciones de costos #####

df['Costo_real_de_la_CS'].corr(df['Costo_ventas_perdidas']) ## mientras aumenta costo real disminuye ventas perdidas
df['Costo_ventas_perdidas'].corr(df['Costo']) ## Las ventas perdidas tienen mayor peso sobre el costo


df.sort_values(['Ventas_perdidas',], ascending=False)



######Arcos faltantes ####################

fig=px.histogram(df, x='Arcos_faltantes',nbins=30)
fig.update_traces(marker=dict(line=dict(width=0.5,color='black')),
                  selector=dict(type='histogram'))
fig.show()

#####logaritmo de ventas perdidas para disminuir sesgo #####

y_nozero=y.replace([0],100)
y_nozero.sort_values()
y_log=np.log(y_nozero)
y_log.info()
#log_vp.isin([np.inf, -np.inf]).value_counts()

#y=df['Ventas_perdidas'].sort_values()
#y.isin([7000000, 25000000]).value_counts()


#y1=y_log.replace([np.inf, -np.inf],6 )

fig=px.histogram( x=y_log,nbins=30)
fig.update_traces(marker=dict(line=dict(width=0.5,color='black')),
                  selector=dict(type='histogram'))
fig.show()

