import numpy as np
import pandas as pd
import sqlite3 as sql
import openpyxl


def arc_clas(url_arc='data/arcsData.txt'):
    ####Leer info de arcos
    df_arc_ff=pd.read_table(url_arc, header=None,sep=" ")
    df_arc_ff.columns=['origen','destino','demanda','prob_fallo']
    df_arc_ff["arc"]=df_arc_ff['origen'] + ' - ' +df_arc_ff['destino']
    
    con=sql.connect("db_arc_nodes_g")
    df_arc_ff.to_sql('info_arc', con, if_exists='replace')
    
    
    
    return(df_arc_ff)

def set_node_df(url_nodes='data/nodesData.txt',url_nodeclas='data/clasificacionArcos.txt'):  
    
    #### leer nombre nodos

    df_nodes=pd.read_table(url_nodes, header=None,sep=" ")
    df_nodes['cod_node']=df_nodes.index
    df_nodes.columns=['name_node','code_node']

    ### leer características nodos

    df_nodes_carc=pd.read_table(url_nodeclas, header=None,sep='&')

    x=df_nodes_carc[0].str.split("/",expand=True)
    x.columns=["Supplier","Plant","CD","Customer"]
    x['Supplier']=x['Supplier'].str.split("[",expand=True)[1]
    x["Supplier"]=x["Supplier"].str.strip()

    y=df_nodes_carc[1].str.split("]",expand=True)
    y1=y[0].str.split("/",expand=True)
    y1.columns=["Supplier","Plant","CD","Customer"]
    y1['Customer']=y1['Customer'].str.strip()
    nodes=y[1].str.split("-",expand=True)
    x_node =nodes[0].str.split("(",expand=True)
    x_node=x_node[1]
    x_node.column=["name_node"]
    y_node=nodes[1].str.split(")", expand=True)[0]

    x["name_node"]=x_node
    y1['name_node']=y_node

    x_y=x.append(y1)

    x_y.replace(['-',' -','-  '],0, inplace=True)
    x_y.replace(['Supplier','Plant','CD','Customer', ' Supplier', 'Customer '],1, inplace=True)
    x_y.drop_duplicates(inplace=True)

    df_nodes= df_nodes.merge(x_y, how='inner')

    df_nodes=df_nodes[['code_node','name_node','Supplier','Plant','CD','Customer']]

    con=sql.connect("db_arc_nodes_g")
    df_nodes.to_sql('info_nodes', con, if_exists='replace')

    return(df_nodes)

def set_arcsce_df(url_arcsce='data/3. fullFlexArcos.txt'):


    df_sce=pd.DataFrame()
    fila_df=pd.DataFrame()
    i=1
    with open(url_arcsce) as f:
        for line in f:
            fila=line.split('[')
            
            if (len(fila)==3):
                sce=fila[0].strip()
                arc_fail=fila[2].replace(']\n','').strip()
                fila_df=pd.DataFrame([[sce,arc_fail]],columns=["escenario",'arc_fail'])
                df_sce=pd.concat([df_sce,fila_df],ignore_index=True) 
            
            if (len(fila)==2):
                
                arc_fail=fila[1].replace(']\n','').strip()
                fila_df=pd.DataFrame([[sce,arc_fail]],columns=["escenario",'arc_fail'])
                df_sce=pd.concat([df_sce,fila_df],ignore_index=True) 
                
    con=sql.connect("db_arcsce_ff")
    df_sce.to_sql("df_arcsce", con)

    return(df_sce)            
                
def set_df_full_arc_sce():
    
    con=sql.connect("db_arcsce_ff")
    cur= sql.Cursor(con)
        
    df_arc_ff= arc_clas()
    df_arc_ff.to_sql('df_all_arcs', con)
    
    cur.execute('''
                create table df_all_sce as
                select distinct escenario
                from df_arcsce
                
                ''')

    cur.execute('''
                create table df_all_arcsce as
                select arc,  escenario
                from df_all_arcs join df_all_sce
                
                ''')   

    cur.execute('''drop table if exists  df_arcsce_count ''')
    
    cur.execute('''
                create table df_arcsce_count as
                select a.arc,  a.escenario, iif(arc_fail is null, 0,1) as arc_fail
                from df_all_arcsce a left join 
                df_arcsce b on a.escenario=b.escenario and a.arc=b.arc_fail         
                ''')        

    #### convertir a wide ####

    df_long_arcsce=pd.read_sql("select * from df_arcsce_count", con)
    df_wide_arcsce=df_long_arcsce.pivot(index='escenario',columns='arc', values='arc_fail')
    df_wide_arcsce.reset_index(inplace=True)
    df_wide_arcsce.to_sql('df_wide_arcsce', con, if_exists="replace", index=False)


    #### depurar base ###
    cur.execute("drop table if exists df_arcsce ")
    cur.execute("drop table if exists df_all_arcs ")
    cur.execute("drop table if exists df_all_sce ")
    cur.execute("drop table if exists df_all_arcsce ")

    cur.execute("vacuum")
    con.close()

##### tablas ####
#### se demora 20  minutos la función set_arcsce_df 

####df_arcsce:  los escenarios con las arcos que fallaron (solo los que fallaron)
##### df_all_arcs:  lista de todos los arcos (144) con información origen destino separa, demanda y prob fallo
#####df_all_sce: lista de todos los escenarios (10080) sin información adicional
#### df_all_arc_sce:  lista de todos los escenarios y todos los arcosa para cada escenario 
####df_arcsce_count: lista de todos los arcos y escenarios con un 1 en los arcos que fallaron y 0 en los que no.
def main():

    arc_clas()
    set_node_df()
    set_arcsce_df()
    set_df_full_arc_sce()
    

main()
