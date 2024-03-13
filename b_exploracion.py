
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

import seaborn as sns

import matplotlib.pyplot as plt

from statistics import linear_regression
from tabnanny import verbose
import streamlit as st


import plotly.graph_objects as go
import matplotlib
import matplotlib.ticker as ticker



import os

os.getcwd()

##################################################################
##################################################################
##################Conectarse BD y revisar tablas##################
##################################################################

### conectarse a bds y unir bases

bds=["data\\db_estFija10","data\\db_estFija20","data\\db_estFija30"]

cons=[]
curs=[]


for i in range(len(bds)):
    print(i)
    print(bds[i])
    con=sql.connect(bds[i]) 
    cur=con.cursor()
    cons.append(con)
    curs.append(cur)


curs[0].execute("select name from sqlite_master where type='table'")
curs[0].fetchall()


####### información de nods y arcos es igual para todos los escenarios

info_arc=pd.read_sql("select * from info_arc", cons[0]).sort_values(by='prob_fallo')
info_nodes=pd.read_sql("select * from info_nodes", cons[0])
info_arc['demanda'].sum()

info_arc.query('prob_fallo==0')
info_arc.columns
info_arc['arc']
#####Tabla con informacion de arcos y kpis


df=pd.read_sql(" select * from kpi_arc_ff", cons[0])
df['esce_prob'] = 'K=1'
df1=pd.read_sql(" select * from kpi_arc_ff", cons[1])
df1['esce_prob'] = 'K=2'
df2=pd.read_sql(" select * from kpi_arc_ff", cons[2])
df2['esce_prob'] = 'K=3'

df_cum=df.append(df1).append(df2)

###validar probabilidades para un arco

arco='Pinchote - Barbosa_Boy'
info_arc[info_arc['arc']==arco]
df[arco].sum()/len(df)
df1[arco].sum()/len(df1)
df2[arco].sum()/len(df2)


### explorar numero de fallos

ax=sns.boxplot(x='esce_prob', y ='Arcos_faltantes', data=df_cum, showmeans=True, palette='viridis')
plt.xlabel('Scenario')
plt.ylabel('Number of closed arc')
plt.show()


group_data=df_cum.groupby('esce_prob')['Arcos_faltantes'].agg(['mean', 'max', 'min']).reset_index()
group_data['mean']=group_data['mean'].round()

#######Explorar variable respuesta ########
sns.set_theme(style="ticks")
f, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(y='esce_prob', x ='Ventas_perdidas', data=df_cum, palette='pastel',showmeans=True, whis=[0, 100], width=.6, meanprops={'marker':'o', 'markerfacecolor':'black'} )
formatter = ticker.FuncFormatter(lambda x, pos: f'{x/1e6:.0f}M')
ax.xaxis.set_major_formatter(formatter)
sns.stripplot(y='esce_prob', x ='Ventas_perdidas', data=df_cum,size=4, color=".3", alpha=.25, palette='dark')
plt.ylabel('Scenario')
plt.xlabel('Number of unsatisfied demand products ($U_{kj}$)')
plt.show()

group_data=df_cum.groupby('esce_prob')['Ventas_perdidas'].agg(['mean', 'max', 'min']).reset_index()
group_data2=group_data[['mean','max', 'min']]/1000000
group_data2["porc_prom_vp"]= group_data2['mean']/group_data2['max']
group_data2[['mean','min','max','porc_prom_vp']]=group_data2[['mean','min','max','porc_prom_vp']].round(2)



####Escenarios de colapos######


max=np.floor(group_data['max'].max())
df_cum['Ventas_perdidas'].value_counts().sort_index()

s_vp=df_cum[df_cum['Ventas_perdidas']==0].groupby('esce_prob')['escenario'].count() ### 626 escenario sin ventas perdidas
colap=df_cum[df_cum['Ventas_perdidas']>=max].groupby('esce_prob')['escenario'].count()### 591 escenarios con colapso de demanda

n_esce=len(df)

df_colapsos= pd.DataFrame({'n_colapsos':colap, 'n_sin_vp': s_vp}).reset_index()
df_colapsos=df_colapsos.fillna(0)
df_colapsos['porc_colpasos']= (df_colapsos['n_colapsos']/n_esce).round(1)
df_colapsos= df_colapsos[['esce_prob', 'n_colapsos', 'porc_colpasos', 'n_sin_vp']]


df_colapsos['Red']=df_colapsos['esce_prob'] +' ('+df_colapsos['porc_colpasos'].astype(str)+')'
ax=sns.barplot(x='Red', y='n_colapsos', data=df_colapsos, color='green')
ax.bar_label(ax.containers[0])
plt.title('Número de colapsos')


###################


X=df_cum.drop(['escenario','Costo','Ventas_perdidas','Arcos_faltantes','Costo_ventas_perdidas','Costo_real_de_la_CS',"Tiempo_tot_esc"],axis=1)
y=df_cum['Ventas_perdidas']


prob_fallo_sim=X.groupby("esce_prob").sum()/n_esce
prob_fallo_sim=prob_fallo_sim.T
prob_fallo_sim.reset_index(inplace=True)
prob_fallo_sim=prob_fallo_sim.rename(columns={'index':'arc'})
prob_fallo_sim.columns.name = None



prob_fallos=prob_fallo_sim.merge(info_arc[['arc','prob_fallo']], how='inner', on= 'arc' ).sort_values('prob_fallo', ascending=False)
prob_fallos[['estFija10','estFija20', 'estFija30']]=prob_fallos[['estFija10','estFija20', 'estFija30']].round(2)
prob_fallos= prob_fallos[['arc','prob_fallo','estFija10','estFija20', 'estFija30']]


prob_fallos2=prob_fallos.query('prob_fallo >0')
np.mean(prob_fallos2['estFija10'])
np.mean(prob_fallos2['estFija20'])
np.mean(prob_fallos2['estFija30'])

0.24*1.2

##################################################################
##################################################################
##################Explorar Datos################################
##################################################################


##### correlaciones de costos #####



df['Ventas_perdidas'].corr(df['Arcos_faltantes']) ## Las ventas perdidas tienen mayor peso sobre el costo
df1['Ventas_perdidas'].corr(df1['Arcos_faltantes']) ## Las ventas perdidas tienen mayor peso sobre el costo
df2['Ventas_perdidas'].corr(df2['Arcos_faltantes']) ## Las ventas perdidas tienen mayor peso sobre el costo


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

ax=sns.scatterplot(x='Arcos_faltantes', y='Ventas_perdidas', palette='viridis',data=df, ax=axes[0])
formatter = ticker.FuncFormatter(lambda x, pos: f'{x/1e6:.0f}M')
axes[0].set_xlabel("K=1 Number of closed arc")
axes[0].set_ylabel("Unsatisfied demand")
ax.yaxis.set_major_formatter(formatter)




ax=sns.scatterplot(x='Arcos_faltantes', y='Ventas_perdidas', color='black',data=df1, ax=axes[1])
formatter = ticker.FuncFormatter(lambda x, pos: f'{x/1e6:.0f}M')
ax.yaxis.set_major_formatter(formatter)
axes[1].set_ylabel("")
axes[1].set_xlabel("K=2 Number of closed arc")




ax=sns.scatterplot(x='Arcos_faltantes', y='Ventas_perdidas', color='grey',data=df2,ax=axes[2])
formatter = ticker.FuncFormatter(lambda x, pos: f'{x/1e6:.0f}M')
ax.yaxis.set_major_formatter(formatter)
axes[2].set_xlabel("K=3 Number of closed arc")
axes[2].set_ylabel("")
