import numpy as np
import pandas as pd
import sqlite3 as sql
import openpyxl
import _funciones as fns





con_kpi=sql.connect("db_kpis") ## para kpis
con_arcsce=sql.connect("db_arcsce") ##para escenarios caracterizado con arcos que fallan
con_arcnode=sql.connect('db_arc_nodes_g') ### para info de arcos y nodos

cur_arcsce=con_arcsce.cursor()
cur_arcnode=con_arcnode.cursor()
cur_kpi=con_kpi.cursor()

### caragar datosde escenarios temporalmente ###


df_ffkpi=pd.read_table('data/4. fullFlexKPI.txt', header=None)
df_ffkpi.columns=['todo']
df_ffkpi.to_sql('df_ffkpi', con_kpi, if_exists="replace")


#### ejecutar sql para seperarcolumnas ####

fns.ejecutar_sql('kpis_to_df.sql', cur_kpi)

###########

df1=pd.read_sql('select* from df_ffkpi_t', con_kpi)
df2=pd.read_sql("select * from df_wide_arcsce", con_arcsce)
df3=df2.merge(df1,how='inner', on='escenario')

df3.to_sql('df_arcsce_kpi', con_arcsce,if_exists='replace')


cur_arcsce.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cur_arcsce.fetchall())


