

drop table if exists df_ffkpi_t;

create table df_ffkpi_t as
select
substr(todo, 1, instr(todo,'- Costo')-2) as escenario,

cast(
    substr(todo, 
    instr(todo,'- Costo =')+10, 
    instr(todo,'- Numero')-(instr(todo,'- Costo')+10))
    as float) as costo,

cast(
    substr(todo, 
    instr(todo,'- Ventas Perdidas')+18, 
    instr(todo,'- Costo Real de la CS')-1-instr(todo,'- Ventas Perdidas')-18) 
    as float) as ventas_perdidas

from df_ffkpi
