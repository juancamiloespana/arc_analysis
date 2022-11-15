from turtle import title
from numpy import expand_dims
import pandas as pd
import matplotlib as mat
import pandas_profiling as pd_prof
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure 



df_ffkpi=pd.read_sql(" select * from kpi_arc_ff", con)
##Análisis descriptivo
df_ffkpi.describe()

#Histogramas
plt.hist(df_ffkpi['Costo'])
#Dispersión y variabilidad
plt.hist(df_ffkpi['Arcos_faltantes'], ec="black")
#está centrado. ¿Se debe a la simulación?
plt.hist(df_ffkpi['Costo_ventas_perdidas'], ec="black")
#Dispersión y variabilidad
plt.hist(df_ffkpi['Ventas_perdidas'], ec="black")
#Dispersión y variabilidad
plt.hist(df_ffkpi['Costo_real_de_la_CS'], ec="black")
#leve sesgo a la derecha
plt.hist(df_ffkpi['Tiempo_tot_esc'], ec="black")

#Gráficos de dispersión
df_ffkpi
sns.scatterplot(data=df_ffkpi, x='Costo_ventas_perdidas', y='Arcos_faltantes') 
sns.scatterplot(data=df_ffkpi, x='Arcos_faltantes', y='Ventas_perdidas')
sns.scatterplot(data=df_ffkpi, x='Arcos_faltantes', y='Costo_real_de_la_CS')

#Parece no haber relación entre los arcos faltantes y el costo de las ventas perdidas
#No es la cantidad de arcos la que afecta las ventas perdidas. Debe haber arcos criticos

sns.scatterplot(data=df_ffkpi, x='Tiempo_tot_esc', y='Arcos_faltantes')
sns.scatterplot(data=df_ffkpi, x='Costo_ventas_perdidas', y='Costo_real_de_la_CS')
sns.scatterplot(data=df_ffkpi, x='Costo_ventas_perdidas', y='Ventas_perdidas')
sns.scatterplot(data=df_ffkpi, x='Ventas_perdidas', y='Costo_real_de_la_CS')
sns.scatterplot(data=df_ffkpi, x='Costo', y='Costo_real_de_la_CS')

#Gráfico de correlación
figure(figsize=(7, 5), dpi=80);
sns.heatmap(df_ffkpi.corr(), annot = True, cmap="Blues",center=0)
#La correlación entre costo y costo de ventas perdidad/ ventas perdidas es muy fuerte
#La correlación entre el costo y el costo real de la CS es inversamente proporcional aligual que las ventas perdidas y el costo de ventas perdidas
#La correlación entre los arcos faltantes y el costo real de la CS es inversamente proporcional, aunque no es muy fuerte
