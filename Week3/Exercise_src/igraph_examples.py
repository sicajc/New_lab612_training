#%%
from igraph import *

#%%
g = Graph(directed = True)
g.add_vertices(5)
#%%

#add ids and labels to vertices
for i in range(len(g.vs)):
    g.vs[i]["id"] = i
    g.vs[i]["label"] = str(i)

#Add edges
g.add_edges([(0,2),(0,1),(0,3),(1,2),(2,4),(3,4)])

#add weights and edges labels
weights = [8,6,3,5,6,4,9]
g.es['weight'] = weights
g.es['label'] = weights
#%%

visual_style = {}
out_name = "graph.png"
# Set bbox and margin
visual_style["bbox"] = (400,400)
visual_style["margin"] = 27
# Set vertex colours
visual_style["vertex_color"] = 'white'
# Set vertex size
visual_style["vertex_size"] = 45
# Set vertex lable size
visual_style["vertex_label_size"] = 22
# Don't curve the edges
visual_style["edge_curved"] = False
# Set the layout
my_layout = g.layout_lgl()
visual_style["layout"] = my_layout
# Plot the graph
plot(g, out_name, **visual_style)
# %%
