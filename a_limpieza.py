import numpy as np
import pandas as pd
import sqlite3 as sql


con=sql.connect("db_arc")
cur= sql.Cursor(con)


df_ffkpi=pd.read_table('data/4. fullFlexKPI.txt', header=None)
df_ffkpi.columns=['todo']
df_ffkpi.to_sql('df_ffkpi', con, if_exists="replace")

df_ffkpi2=pd.read_sql('''
                      
select
substr(todo, 11, instr(todo,'- Costo')-11) as escenario,
substr(todo, instr(todo,'- Costo =')+10, instr(todo,'- Numero')-(instr(todo,'- Costo')+10)) as costo

from df_ffkpi''', con)

df_ffkpi2.info()
