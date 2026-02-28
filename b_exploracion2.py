
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
#import streamlit as st


import plotly.graph_objects as go
import matplotlib
import matplotlib.ticker as ticker



import os
from os import listdir
os.getcwd()

##################################################################
##################################################################
##################Conectarse BD y revisar tablas##################
##################################################################

### conectarse a bds y unir bases
paths_dbs=['data/DB1', 'data/DB2']

ventas_perdidas_cum=pd.DataFrame()
arcos_faltantes_sum_cum=pd.DataFrame()
sum_colapsos_cum=pd.DataFrame()

for path in paths_dbs:
    list_dbs=os.listdir(path)
    
    for db in list_dbs:
        print(f'Inicio procesamiento: {db} en path: {path}')

        con=sql.connect(path+'/'+db)
        cur=con.cursor()

        cur.execute("select name from sqlite_master where type='table'")
        cur.fetchall()



        df_cum=pd.read_sql(" select * from kpi_arc_ff", con)

        # resumen ventas perdidas
        ventas_perdidas=df_cum['Ventas_perdidas'].agg(['mean', 'max', 'min']).reset_index()
        ventas_perdidas['escenario']=db[3:]
        ventas_perdidas_cum=pd.concat([ventas_perdidas_cum, ventas_perdidas], ignore_index=True)    

        
        ### cantidad de colapsos por escenario
        max= ventas_perdidas['Ventas_perdidas'][1]

        sum_colapsos=pd.read_sql(""" select 
                                 count(iif(Ventas_perdidas >= """+str(max)+""", 1, null)) as count_colapsos,
                                 count(iif(Ventas_perdidas == 0, 1, null)) as count_full_demand
                                 from kpi_arc_ff  """, con)
        sum_colapsos['escenario']=db[3:]
        sum_colapsos_cum=pd.concat([sum_colapsos_cum, sum_colapsos], ignore_index=True)

        # resumen arcos faltantes
        arcos_faltantes=pd.read_sql(""" select escenario, sum(arc_fail) as count_failures from df_arcsce_count group by escenario  """, con)
        arcos_faltantes_sum=arcos_faltantes['count_failures'].agg(['mean', 'max', 'min']).reset_index()
        arcos_faltantes_sum['escenario']=db[3:]
        arcos_faltantes_sum_cum=pd.concat([arcos_faltantes_sum_cum, arcos_faltantes_sum], ignore_index=True)    


path_results='resultados/exploracion/'
ventas_perdidas_cum[['escenario','Ventas_perdidas','index']].to_excel(path_results+'ventas_perdidas.xlsx', index=False)
arcos_faltantes_sum_cum[['escenario','count_failures','index']].to_excel(path_results+'arcos_faltantes.xlsx', index=False)
sum_colapsos_cum[['escenario','count_colapsos','count_full_demand']].to_excel(path_results+'sum_colapsos.xlsx', index=False)