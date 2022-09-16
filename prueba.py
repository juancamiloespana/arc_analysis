from numpy import expand_dims
import pandas as pd

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
df_ffkpi[0] = df_ffkpi[0].apply(lambda x : int(x[10:-1])) #quité escenario
df_ffkpi[1] = df_ffkpi[1].apply(lambda x : (x[9:]))
df_ffkpi[1] = df_ffkpi[1].apply(lambda x : (x[:-1]))
df_ffkpi[2] = df_ffkpi[2].apply(lambda x : (x[28:-1]))
df_ffkpi[3] = df_ffkpi[3].apply(lambda x : (x[23:-1]))
df_ffkpi[4] = df_ffkpi[4].apply(lambda x : (x[17:-1]))
df_ffkpi[5] = df_ffkpi[5].apply(lambda x : (x[21:-1]))
df_ffkpi[6] = df_ffkpi[6].apply(lambda x : (x[35:])) #no sigue un guón


df_ffkpi.set_index(0, inplace = True)
df_ffkpi = df_ffkpi.rename(columns={0:'Escenario',1:'Costo', 2:'Arcos_faltantes',3:'Costo_ventas_perdidas',4:'Ventas_perdidas',5:'Costo_real_de_la_CS',6:'Tiempo_tot_esc'})
#df_ffkpi[Costo] = df_ffkpi[Costo].apply(lambda x : float(x[10:]))

#df_ffkpi['Costo'].describe()
#df_ffkpi.plot(kind='Arcos faltantes')
print(df_ffkpi)
#print("Costo máximo: "+df_ffkpi['Costo'].max())
#print("Costo mínimo: "+df_ffkpi['Costo'].min())
#print("Costo promedio: "+df_ffkpi['Costo'].mean())
#shift+enter para correr a la derecha



