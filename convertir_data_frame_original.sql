

drop table if exists df_ffkpi_t;

create table df_ffkpi_t as
select
substr(todo, 11, instr(todo,'- Costo')-11) as escenario,
substr(todo, instr(todo,'- Costo =')+10, instr(todo,'- Numero')-(instr(todo,'- Costo')+10)) as costo

from df_ffkpi
