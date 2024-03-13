from geopy.geocoders import Nominatim


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

#### para mapas

import folium as fo



bds=["data\\db_estFija10","data\\db_estFija20","data\\db_estFija30"]

con=sql.connect(bds[0]) 
cur=con.cursor()

cur.execute("select name from sqlite_master where type='table'")
cur.fetchall()

##################Cargar coordenadas

coord=pd.read_csv('data\\coordenadas\\Coordenadas.csv')
coord.to_sql('coordenadas', con, if_exists="replace")
####### información de nods y arcos es igual para todos los escenarios

info_arc=pd.read_sql("select * from info_arc", con).sort_values(by='prob_fallo')
info_nodes=pd.read_sql("""with t1 as ( 
                       select 
                       code_node,
                       case when name_node = 'Bogota1' then 'Bogota'
                        when name_node = 'Bogota2' then 'Bogota'
                        when name_node = 'Barranquilla1' then 'Barranquilla'
                        when name_node = 'Ibague1' then 'Ibague'
                        when name_node = 'Medellin1' then 'Medellin'
                        when name_node = 'Pereira1' then 'Pereira' else name_node
                        end as name_node,
                        Supplier,
                        Plant,
                        CD,
                        Customer
                        from info_nodes)
                        select 
                        a.*, b.Latitude as latitude, 
                        b.Longitude as longitude  
                        from t1 a left join 
                        coordenadas b 
                        on a.name_node =b.Name """, con)





#################info arcos
info_arc['origen'].replace('_',' ', inplace=True)
info_arc['destino'].replace('_',' ', inplace=True)


list_lat_o=[]
list_long_o=[]
list_lat_d=[]
list_long_d=[]

for nod in range(len(info_arc)):
    origen=info_arc.iloc[nod,1]
    destino=info_arc.iloc[nod,2]
    lat_o, long_o = get_coordinates(origen)
    lat_d, long_d = get_coordinates(destino)
    list_lat_o.append(lat_o)
    list_long_o.append(long_o)
    list_lat_d.append(lat_d)
    list_long_d.append(long_d)
    
    

info_arc['lat_o']= list_lat_o
info_arc['long_o']= list_long_o
info_arc['lat_d']= list_lat_d
info_arc['long_d']= list_long_d

len(list_lat_o)
len(info_arc)

info_arc.isna().sum()

info_arc2=info_arc.dropna()


###############################



# Create a base map
my_map = folium.Map(location=[info_nodes.iloc[2,7], info_nodes.iloc[2,8]], zoom_start=4)

# Add markers for each location
for location in info_nodes.index:
    fo.CircleMarker([info_nodes.iloc[location,7], info_nodes.iloc[location,8]], popup=info_nodes.iloc[location,2],
             radius=5).add_to(my_map)
    


for location in range(len(info_arc2)):
    coord_o = [info_arc2.iloc[location,6],info_arc2.iloc[location,7] ]
    coord_d = [info_arc2.iloc[location,8],info_arc2.iloc[location,9] ]
    fo.PolyLine(
                [coord_o, coord_d],
                color="green",
                weight=1,
                opacity=1).add_to(my_map)
    

    
    


# Save the map as an HTML file or display it
my_map.save("map_with_points.html")

