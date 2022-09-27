from turtle import title
from numpy import expand_dims
import pandas as pd
import matplotlib as mat
import pandas_profiling as pd_prof
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure 


#import regex as re
df_ffkpi=pd.read_table('data/4. fullFlexKPI.txt', header=None) #leer los datos
#print(df_ffkpi)
#df_ffkpi.str.split('-',extend=True)
#datos1 = df_ffkpi.split(" ")
#Escenario = datos[2] #id de cada escenario
#Costo1= datos[6]
#Costo2=Costo1[:-3]
df_ffkpi = df_ffkpi[0].str.split('-', expand=True)
#Funcion lambda funciona como un for que itera y elimina en cada registro o fila los caracteres que le indique
df_ffkpi[0] = df_ffkpi[0].apply(lambda x : (x[:-1])) #quité escenario
df_ffkpi[1] = df_ffkpi[1].apply(lambda x : (x[9:]))
df_ffkpi[1] = df_ffkpi[1].apply(lambda x : (x[:-1]))
df_ffkpi[2] = df_ffkpi[2].apply(lambda x : (x[28:-1]))
df_ffkpi[3] = df_ffkpi[3].apply(lambda x : (x[23:-1]))
df_ffkpi[4] = df_ffkpi[4].apply(lambda x : (x[17:-1]))
df_ffkpi[5] = df_ffkpi[5].apply(lambda x : (x[21:-1]))
df_ffkpi[6] = df_ffkpi[6].apply(lambda x : (x[35:])) #no sigue un guón


df_ffkpi.set_index(0, inplace = True)
df_ffkpi = df_ffkpi.rename(columns={0:'Escenario',1:'Costo', 2:'Arcos_faltantes',3:'Costo_ventas_perdidas',4:'Ventas_perdidas',5:'Costo_real_de_la_CS',6:'Tiempo_tot_esc'})

df_ffkpi.info()
df_ffkpi=df_ffkpi.astype({'Costo':float,'Arcos_faltantes':int,'Costo_ventas_perdidas':float,'Ventas_perdidas':float,'Costo_real_de_la_CS':float,'Tiempo_tot_esc':float}) #cambiar tipo de dato a numerico
##Análisis descriptivo
df_ffkpi.describe()

#Histogramas
plt.hist(df_ffkpi['Costo'])
plt.hist(df_ffkpi['Arcos_faltantes'], ec="black")
plt.hist(df_ffkpi['Costo_ventas_perdidas'], ec="black")
plt.hist(df_ffkpi['Ventas_perdidas'], ec="black")
plt.hist(df_ffkpi['Costo_real_de_la_CS'], ec="black")
plt.hist(df_ffkpi['Tiempo_tot_esc'], ec="black")

#Gráficos de dispersión
df_ffkpi
sns.scatterplot(data=df_ffkpi, x='Costo_ventas_perdidas', y='Arcos_faltantes') 
sns.scatterplot(data=df_ffkpi, x='Arcos_faltantes', y='Costo_ventas_perdidas')
sns.scatterplot(data=df_ffkpi, x='Arcos_faltantes', y='Ventas_perdidas')
#Parece no haber relación entre los arcos faltantes y el costo de las ventas perdidas
#No es la cantidad de arcos la que afecta las ventas perdidas. Debe haber arcos criticos

sns.scatterplot(data=df_ffkpi, x='Tiempo_tot_esc', y='Arcos_faltantes')
sns.scatterplot(data=df_ffkpi, x='Costo_ventas_perdidas', y='Costo_real_de_la_CS')
sns.scatterplot(data=df_ffkpi, x='Costo_ventas_perdidas', y='Ventas_perdidas')
sns.scatterplot(data=df_ffkpi, x='Ventas_perdidas', y='Costo_real_de_la_CS')
figure(figsize=(7, 5), dpi=80);
sns.heatmap(df_ffkpi.corr(), annot = True, cmap="Blues",center=0)