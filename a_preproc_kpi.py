from numpy import expand_dims
import pandas as pd
import sqlite3 as sql


def prepro_kpi(ruta='data/4. fullFlexKPI.txt'):
    #import regex as re
    df_ffkpi=pd.read_table(ruta, header=None) #leer los datos
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
    #df_ffkpi[Costo] = df_ffkpi[Costo].apply(lambda x : float(x[10:]))

    #df_ffkpi['Costo'].describe()
    #df_ffkpi.plot(kind='Arcos faltantes')
    print(df_ffkpi)
    #print("Costo máximo: "+df_ffkpi['Costo'].max())
    #print("Costo mínimo: "+df_ffkpi['Costo'].min())
    #print("Costo promedio: "+df_ffkpi['Costo'].mean())
    #shift+enter para correr a la derecha
    
    return(df_ffkpi)



df=prepro_kpi(ruta='data/4. fullFlexKPI.txt')
df2=prepro_kpi(ruta='data/2. esp2ProdKPI.txt')



con=sql.connect('db_kpis') ## crea o se conecta a base de datos
cur=con.cursor() ## para ejecutar sql en base de datos

#con2=sql.connect('db_arcsce') ### se puede conectar a otras base de datos

df.to_sql('kpi_ff', con) ## para llevar tabla pandas a bd
df2.to_sql('kpi_esp2prod', con)


### para traer base de datos

#df=pd.read_sql("select * from kpi_ff", con)


#cur.execute("create table suma_costo as select sum(Costo) as suma from kpi_ff")

#pd.read_sql("select * from suma_costo", con)

#cur.execute("drop table suma_costo") ## para borrar tabla


#cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
#print(cur.fetchall())