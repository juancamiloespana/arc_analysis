import numpy as np
import pandas as pd
import sqlite3 as sql
from sklearn.feature_selection import VarianceThreshold

### conectarse a bd
con=sql.connect("db_arcsce") 
cur=con.cursor()

cur.execute("select name from sqlite_master where type='table'")
cur.fetchall()

##### cargar base de datos ####
df=pd.read_sql(" select * from kpi_arc_ff", con)
df

#### separar x y y y eliminar arcos que no cambian

X=df.drop(['escenario','Costo','Ventas_perdidas','Arcos_faltantes','Costo_ventas_perdidas','Costo_real_de_la_CS',"Tiempo_tot_esc"],axis=1)
y=df['Ventas_perdidas']

sel=VarianceThreshold()
X2=sel.fit_transform(X)
colum_out=sel.get_feature_names_out()
X2=pd.DataFrame(X2, columns=colum_out)
X2.info(verbose=True)

#### y para clasificacion
y_clas= (y>7000000) ### donde se da una brecha
y_clas.astype(int)
y_clas.value_counts()

### calcular probabilidad de fallo general
y_clas[y_clas==True].count()/y_clas.count()


##### interacciones ######

inter1=np.equal([X2[X2.columns[88]] + X2[X2.columns[31]]] ,2)[0]
np.unique(inter1, return_counts=True)

inter2=np.equal([X2[X2.columns[88]] + X2[X2.columns[58]] ],2)[0]
np.unique(inter2, return_counts=True)

inter3=np.equal([X2[X2.columns[51]] + X2[X2.columns[59]]],2)[0]
np.unique(inter3, return_counts=True)

inter4=np.equal([X2[X2.columns[57]] + X2[X2.columns[51]] + X2[X2.columns[53]]],3)[0]
np.unique(inter4, return_counts=True)

inter5=np.equal([X2[X2.columns[88]] + X2[X2.columns[31]] + X2[X2.columns[96]]],3)[0]
np.unique(inter5, return_counts=True)


X2['inter1']=inter1
X2['inter2']=inter2
X2['inter3']=inter3
X2['inter4']=inter4
X2['inter5']=inter5


def dif_prop(var_name):
    
    arcs=X2[var_name]
    nc_aok=np.count_nonzero((y_clas==0) & (arcs==0))
    c_aok=np.count_nonzero((y_clas==1) & (arcs==0))
    aok=np.count_nonzero(arcs==0)
    prop_col_aok = c_aok/aok

    nc_af=np.count_nonzero((y_clas==0) & (arcs==1))
    c_af=np.count_nonzero((y_clas==1) & (arcs==1))
    af=np.count_nonzero(arcs==1)
    prop_col_af = c_af/af

    prop_dif =prop_col_af -prop_col_aok 
     
    return(prop_col_af, prop_col_aok , prop_dif, aok, af)




def todos_arcs():
    
    arc_list=[]
    prop_dif_list=[]
    aok_list=[]
    af_list=[]
    prop_col_af_list=[]
    prop_col_aok_list=[]
    

    
    for var_n in X2.columns:
    
        prop_col_af, prop_col_aok, prop_dif, aok, af = dif_prop(var_n)
        arc_list.append(var_n)
        prop_dif_list.append(prop_dif)
        aok_list.append(aok)
        af_list.append(af)
        prop_col_af_list.append( prop_col_af)
        prop_col_aok_list.append( prop_col_aok)
            
    
    return( prop_col_af_list, prop_col_aok_list,arc_list, prop_dif_list, aok_list,af_list)
        



prop_col_af_list, prop_col_aok_list,arc_list, prop_dif_list, aok_list,af_list = todos_arcs()

var_n=arc_list[1]

dif=pd.DataFrame({'arc':arc_list,'prop_col_af': prop_col_af_list, 'prop_col_aok':prop_col_aok_list,'prop_dif': prop_dif_list, 'n_sce_aok':aok_list,'n_sce_af':af_list})


dif.sort_values('n_sce_af', ascending=False)

dif.to_excel('resultados\\dif_arcos.xlsx')



dif_prop('inter1')