from statistics import linear_regression
import streamlit as st
import pandas as pd
import sqlite3 as sql
import plotly.express as px
import plotly.graph_objects as go

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

from sklearn.ensemble import RandomForestRegressor



con_arcsce=sql.connect("db_arc_nodes_g") ##para escenarios caracterizado con arcos que fallan
cur_arcsce=con_arcsce.cursor()


pd.read_sql("select * from Info_nodes", con_arcsce)

df=pd.read_sql("select * from df_arcsce_kpi", con_arcsce)
#st.write("Helllo world")
#st.altair_chart(df['ventas_perdidas'])


fig=px.histogram(df, x='ventas_perdidas',nbins=30)
fig.update_traces(marker=dict(line=dict(width=0.5,color='black')),
                  selector=dict(type='histogram'))
fig.show()

fig=px.histogram(df, x='costo')
fig.show()




X=df.drop(['index','escenario','costo','ventas_perdidas'],axis=1)
y=df['ventas_perdidas']

reg=LinearRegression()
RF=RandomForestRegressor(n_estimators=500)
RF.fit(X,y)
reg.fit(X,y)

coef=reg.coef_
name=reg.feature_names_in_

df_coef=pd.DataFrame()
df_coef['arc']=name
df_coef['coef']=coef

df_coef.sort_values(by='coef', ascending=False)
df_coef.sort_values(by='coef', ascending=True)
y_pred=reg.predict(X)
mean_absolute_percentage_error(y,y_pred)


y_pred=RF.predict(X)
mean_absolute_percentage_error(y,y_pred)

max(y-y_pred)

res=pd.DataFrame(y/1000)
res['pred']=y_pred/1000