import sqlite3 as sql
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt




con=sql.connect("data\\db_estFija10")
cur=con.cursor()

cur.execute("select name from sqlite_master where type='table'")
cur.fetchall()


info_arc=pd.read_sql("select* from info_arc", con)
info_nodes=pd.read_sql("select* from info_nodes", con)
info_node=pd.read_sql("""
                      select a.*, b.Supplier, b.Plant, b.CD,b.Customer 
                      from coordenadas a left join info_nodes b on
                      b.name_node=a.Name """, con)

#pd.read_sql("select * from info_nodes where name_node like '%Bogota%' ", con)

info_node.drop(columns=['Group','index'], inplace =True)

info_nodes2=info_node.drop_duplicates(keep='first')
info_nodes2.reset_index(inplace=True, drop=True)

info_arc[info_arc['destino'].str.contains('2', regex=False)]

info_arc['destino']=info_arc['destino'].str.replace('2', '', regex=False)


G = nx.from_pandas_edgelist(info_arc, source='origen', target='destino', edge_attr='demanda', create_using=nx.Graph())

#G.nodes['Caucasia']['Longitude']

#node='Caucasia'

for _, row in info_nodes2.iterrows():

    G.nodes[row['Name']]['Latitude'] = row['Latitude']  
    G.nodes[row['Name']]['Longitude'] = row['Longitude']
    G.nodes[row['Name']]['Supplier'] = row['Supplier']
    G.nodes[row['Name']]['Plant'] = row['Plant']
    G.nodes[row['Name']]['CD'] = row['CD']
    G.nodes[row['Name']]['Customer'] = row['Customer']
    print(row)
    
# for node in G.nodes():
#     print(node)
#     G.nodes[node]['Latitude']
#G.nodes['Villavicencio']

degree_centrality = nx.degree_centrality(G)
pos = {node: (G.nodes[node]['Longitude'], G.nodes[node]['Latitude']) for node in G.nodes()}

plt.figure(figsize=(8, 6))

# Draw the graph with customized node size, node color, and labels

nx.draw(G, pos, with_labels=False, node_size=[v * 1000 for v in degree_centrality.values()],
        node_color='skyblue', edge_color='gray', font_size=10, font_weight='bold')

# Draw edge labels for weights
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

# Display the graph
plt.title('Graph Visualization with NetworkX')
plt.show()


# General metric
edge_betweenness_centrality = nx.edge_betweenness_centrality(G, normalized=False)

# Edge Load
edge_load = nx.edge_load_centrality(G)

cf_bc=nx.edge_current_flow_betweenness_centrality(G)


#############


 ##############metrics by echelon ##########
 


# subset1:  source Supplier == 1) and target nodes (Plant == 1)
source_nodes = [node for node, data in G.nodes(data=True) if data.get('Supplier') == 1]
target_nodes = [node for node, data in G.nodes(data=True) if data.get('Plant') == 1]

# Calculate edge betweenness centrality using the source and target nodes
edge_betweenness_supplier_plant = nx.edge_betweenness_centrality_subset(G, sources=source_nodes, targets=target_nodes)


# Subset 2: Source = Plant, Target = CD
source_plant_nodes = [node for node, data in G.nodes(data=True) if data.get('Plant') == 1]
target_cd_nodes = [node for node, data in G.nodes(data=True) if data.get('CD') == 1]

# Calculate edge betweenness centrality for Source = Plant and Target = CD
edge_betweenness_plant_cd = nx.edge_betweenness_centrality_subset(G, sources=source_plant_nodes, targets=target_cd_nodes)

# Subset 2: Source = CD, Target = Customer
source_cd_nodes = [node for node, data in G.nodes(data=True) if data.get('CD') == 1]
target_customer_nodes = [node for node, data in G.nodes(data=True) if data.get('Customer') == 1]

# Calculate edge betweenness centrality for Source = CD and Target = Customer
edge_betweenness_cd_customer = nx.edge_betweenness_centrality_subset(G, sources=source_cd_nodes, targets=target_customer_nodes)


### combinar metrics


edge_betweenness_combined = {}
for edge in set(edge_betweenness_supplier_plant) | set(edge_betweenness_plant_cd) | set(edge_betweenness_cd_customer):
    avg_centrality = (
        edge_betweenness_supplier_plant.get(edge, 0) +
        edge_betweenness_plant_cd.get(edge, 0) +
        edge_betweenness_cd_customer.get(edge, 0)
    ) / 3  # Average of the three subsets
    edge_betweenness_combined[edge] = avg_centrality

# Convert the result into a DataFrame
df_edge_betweenness = pd.DataFrame(
    [(edge, centrality) for edge, centrality in edge_betweenness_combined.items()],
    columns=['Edge', 'Betweenness Centrality']
)


# Sort the DataFrame by centrality to identify the most important edges
df_edge_betweenness = df_edge_betweenness.sort_values(by='Betweenness Centrality', ascending=False)









# Combine into a DataFrame for easy viewing
edge_metrics_df = pd.DataFrame({
    'Source': [edge[0] for edge in G.edges()],
    'Target': [edge[1] for edge in G.edges()],
    'Betweenness Centrality': [edge_betweenness_centrality[edge] for edge in G.edges()],
    'Load Centrality': [edge_load[edge] for edge in G.edges()]
    #'Current Flow Betweenes Centrality': [cf_bc[edge] for edge in G.edges()]
    
})

import openpyxl

edge_metrics_df.to_excel('resultados/graph_metrics.xlsx')
edge_metrics_df.sort_values('Betweenness Centrality', ascending=0, inplace=True)
edge_metrics_df.sort_values('Load Centrality', ascending=0)

source=edge_metrics_df['Source'][0:2].to_list()
targe=edge_metrics_df['Target'][0:2].to_list()

edge_betweenness_centrality_s=nx.edge_betweenness_centrality_subset(G,source, targe )


