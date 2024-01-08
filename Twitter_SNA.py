#install necessary libraries
#pip install snscrape
#pip install swifter
#import libraries
import pandas as pd
import numpy as np
import snscrape.modules.twitter as sntwitter
import requests
from requests import get
from requests.exceptions import MissingSchema, InvalidSchema, ConnectionError 
from urllib.parse import urlparse
import swifter
import networkx as nx
from networkx.algorithms import bipartite
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
import community
from community import community_louvain
from collections import Counter
from networkx.algorithms.community import k_clique_communities
from infomap import Infomap
# ### Helpers
#function to convert url from scraping to domains
def get_domain(source):
    
    try :
        ecco = requests.get(source)
        url = ecco.url
        domain = urlparse(url).netloc
    except (MissingSchema,ConnectionError,InvalidSchema):
        domain = None
    
    return domain
#scrape tweets using snstwitter 
query = "(war,ukraine,russia) lang:en until:2022-06-31 since:2022-05-31 filter:links -filter:retweets"
tweets = []

limit =  200 # how many tweets which you wanna extract

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
#     print(vars(tweet))
#     break
    try:
        if len(tweets) == limit:
            break
        else:
            tweets.append([tweet.id,tweet.tcooutlinks, tweet.user.displayname])
    except:
        pass
        
#function to get bipartite graph from data
def get_bipartite(data):
    G = nx.Graph()
    id_taken = []
    #for every row in data
    for element in data:

        #if the tweets has been already examined i do nothing
        if element[0] in id_taken:
            
            pass
        #otherwise i have to add it to the list
        else:
            
            id_taken.append(element[0])

            #if i already have a link between source and user
            if G.has_edge(element[1], element[2]):

                #change weight
                weight= G[element[1]][element[2]]['weight'] 
                newweight = int(weight) +1 

                G[element[1]][element[2]]['weight'] = newweight


            else:

                #i add nodes
                #if i already have the source nothing happens
                G.add_node(element[1],bipartite=0, type = 'source', source_name = element[1])
                #
                #as an attribute of the nodes i also put the source shared
                G.add_node(element[2], bipartite=1, type = 'user', source_shared = element[1])

                #add edge with weight 1
                G.add_edge(element[1], element[2], weight=1)

    return G

#function that takes list of nodes and eliminates the ones that don't have links
def get_list(starting_list):
    removed = 0
    for i in list(starting_list.nodes):
        if starting_list.degree(i) < 1:
            #comando per cancellare i
            starting_list.remove_node(i)
            removed += 1
    
    return starting_list


# #### Little Data Processing

#merging all files from scraping
#filenames = ['df10.csv','df7.csv', 'df4.csv','df11.csv','df13.csv','df14.csv','df16.csv','df18.csv','df5.csv','df6.csv','df8.csv','df9.csv','df1.csv']

#df_concat = pd.concat([pd.read_csv(f,sep=';') for f in filenames],ignore_index =True)
#get it into a format to apply get_bipartite()
#df =df_concat[['id','source','user']]
#no external source(the source is citing itself)
#df = df.dropna(how='any',axis=0)
#i rule out retweets
#df = df.drop(df[df['source']=='twitter.com'].index)

#save final csv 
#final_csv = df.to_csv('C:\\Users\\elens\\OneDrive\\Documents\\social network\\final2.csv', sep=';', encoding='utf-8')
df = read_csv('final2.csv',)
df['id'] = df['id'].astype(str)
df['Col 4'] = np.array(df.to_numpy()).tolist()
df_new = df['Col 4']

G = get_bipartite(df_new)
G.nodes()
G.edges()

nx.info(G)
#get total sources
nodi_fonte = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}
len(nodi_fonte)
#get total users
nodi_utente = set(G) - nodi_fonte
len(nodi_utente)
#save graph
nx.write_gexf(G,'Total_graph.gexf')

# ### Density:
# The density is the ratio of number of links to the number of possible links in a nertwork with N nodes. It is a global property of the network since is related to the number of links of the whole network.
density(G)
#get density source nodes
bipartite.density(G, nodi_fonte)
#get density user nodes
bipartite.density(G, nodi_utente)

# ### Degree:
# The degree is the number of links connected to a node so it is a local property of the node.
#get degree of Sources and Users
deg_FONTI, deg_UTENTI = bipartite.degrees(G, nodi_utente)
fonti_degree=list(dict(deg_FONTI).values())
utente_degree = list(dict(deg_UTENTI).values())

# ### Complete Graph
# 
# A complete graph is a graph in which each pair of graph vertices is connected by an edge.

#comparison with complete graph
C = bipartite.complete_bipartite_graph(1142, 2870)
C_nodi_fonte = {n for n, d in C.nodes(data=True) if d["bipartite"] == 0}
C_nodi_utente = set(C) - C_nodi_fonte

#density
bipartite.density(C, C_nodi_utente)


# ### Distribution of the Source Degree
#get basic statistics relating to source degree
print('media: ', np.mean(fonti_degree))
print('mediana: ',np.median(fonti_degree)) 
print('dev st: ',np.std(fonti_degree))
print('massimo: ',np.max(fonti_degree))
print('minimo: ',np.min(fonti_degree))

#to get the hub nodes with degree hiher than 95% of the others
percentile_99 = np.percentile(fonti_degree,99)
print(percentile_99)

hub_nodi = sorted([k for k,v in dict(deg_FONTI).items() if v>= percentile_99])
print(hub_nodi)

# ECDF in linear scale
cdf_function = ECDF(fonti_degree)
x = np.unique(fonti_degree)
y = cdf_function(x)
fig_cdf_function = plt.figure(figsize=(8,5)) 
axes = fig_cdf_function.gca()
axes.plot(x,y,color = 'red', linestyle = '--', marker= 'o',ms = 16)
axes.set_xlabel('Degree',size = 30)
axes.set_ylabel('ECDF',size = 30)

# ECDF in loglog scale
fig_cdf_function = plt.figure(figsize=(8,5))
axes = fig_cdf_function.gca()
axes.loglog(x,y,color = 'red', linestyle = '--', marker= 'o',ms = 16)
axes.set_xlabel('Degree',size = 30)
axes.set_ylabel('ECDF',size = 30)

# ECCDF in loglog scale
y = 1-cdf_function(x)
fig_ccdf_function = plt.figure(figsize=(8,5))
axes = fig_ccdf_function.gca()
axes.loglog(x,y,color = 'red', linestyle = '--', marker= 'o',ms = 16)
axes.set_xlabel('Degree',size = 30)
axes.set_ylabel('ECCDF',size = 30)


# ### Comparison with Random Network for Source Degree
# 
# The random network represents the conventional reference point(null model), since the comparison can be useful to distinguish interesting from non-interesting.
#generate bipartite random graph with same n of nodes, edges and p=density of graph
random_bipartite = bipartite.random_graph(1142, 2870, 0.0009986148147696139)
C_nodi_fonte_random = {n for n, d in random_bipartite.nodes(data=True) if d["bipartite"] == 0}
len(C_nodi_fonte_random)
C_nodi_utente_random = {n for n, d in random_bipartite.nodes(data=True) if d["bipartite"] == 1}
len(C_nodi_utente_random)
#get degrees 
deg_FONTI_RANDOM, deg_UTENTI_RANDOM = bipartite.degrees(random_bipartite, C_nodi_utente_random)
fonti_degree_random = list(dict(deg_FONTI_RANDOM).values())
cdf_reale = ECDF(fonti_degree)
x_reale = np.unique(fonti_degree)
y_reale = cdf_reale(x_reale)

cdf_random = ECDF(fonti_degree_random)
x_random = np.unique(fonti_degree_random)
y_random = cdf_random(x_random)

fig_cdf_reale = plt.figure(figsize=(16,9))

assi = fig_cdf_reale.gca()
assi.set_xscale('log')
assi.set_yscale('log')

assi.loglog(x_reale,1-y_reale,marker='o',ms=8, linestyle='--')
assi.loglog(x_reale,1-y_reale,marker='o',ms=8, linestyle='--')
assi.plot(x_random,1-y_random,marker='+',ms=10, linestyle='--')
assi.set_xlabel('Degree',size=30)
assi.set_ylabel('ECCDF', size = 30)


# ### Distribution of User Degree
#get basic statistics relating to user degree
print('media: ', np.mean(utente_degree))
print('mediana: ',np.median(utente_degree)) 
print('dev st: ',np.std(utente_degree))
print('massimo: ',np.max(utente_degree))
print('minimo: ',np.min(utente_degree))
#to get the hub nodes with degree higher than 95% of the others
percentile_99 = np.percentile(utente_degree,99)
print(percentile_99)

#get user hub nodes
hub_nodi = sorted([k for k,v in dict(deg_UTENTI).items() if v>= percentile_99])
print(hub_nodi)
# ECDF in linear scale
cdf_function = ECDF(utente_degree)
x = np.unique(utente_degree)
y = cdf_function(x)
fig_cdf_function = plt.figure(figsize=(8,5)) 
axes = fig_cdf_function.gca()
axes.plot(x,y,color = 'red', linestyle = '--', marker= 'o',ms = 16)
axes.set_xlabel('Degree',size = 30)
axes.set_ylabel('ECDF',size = 30)

# ECDF in loglog scale
fig_cdf_function = plt.figure(figsize=(8,5))
axes = fig_cdf_function.gca()
axes.loglog(x,y,color = 'red', linestyle = '--', marker= 'o',ms = 16)
axes.set_xlabel('Degree',size = 30)
axes.set_ylabel('ECDF',size = 30)

# ECCDF in loglog scale
y = 1-cdf_function(x)
fig_ccdf_function = plt.figure(figsize=(8,5))
axes = fig_ccdf_function.gca()
axes.loglog(x,y,color = 'red', linestyle = '--', marker= 'o',ms = 16)
axes.set_xlabel('Degree',size = 30)
axes.set_ylabel('ECCDF',size = 30)


# ### Comparison with Random Network
utenti_degree_random = list(dict(deg_UTENTI_RANDOM).values())

cdf_reale = ECDF(utente_degree)
x_reale = np.unique(utente_degree)
y_reale = cdf_reale(x_reale)

cdf_random = ECDF(utenti_degree_random)
x_random = np.unique(utenti_degree_random)
y_random = cdf_random(x_random)

fig_cdf_reale = plt.figure(figsize=(16,9))

assi = fig_cdf_reale.gca()
assi.set_xscale('log')
assi.set_yscale('log')

assi.loglog(x_reale,1-y_reale,marker='o',ms=8, linestyle='--')
assi.loglog(x_reale,1-y_reale,marker='o',ms=8, linestyle='--')
assi.plot(x_random,1-y_random,marker='+',ms=10, linestyle='--')
assi.set_xlabel('Degree',size=30)
assi.set_ylabel('ECCDF', size = 30)


# ### Analize Closeness Centrality, Degree Centrality and Betweenness for Source Nodes

# #### Degree Centrality
# Degree centrality ranks nodes with more connections higher in terms of centrality(is the number of adjacent edges)
degree_centrality = bipartite.degree_centrality(G, nodi_fonte)


# #### Closeness Centrality
# The intuition behind Closeness Centrality is that influential and central nodes can quickly reach other nodes; these nodes should have a smaller average shortest path lenght to other nodes
closeness_centrality = bipartite.closeness_centrality(G, nodi_fonte)


# #### Betweenness Centrality
# It was introduced since Degree Centrality wasn't able to capture all different aspects of the concept of centrality.The idea behind it is to measure the extent to which a node lies on paths betweens other nodes,
# which means that nodes with high betweenness centrality have  control ver information flowing in the network.
betweenness_centrality = bipartite.betweenness_centrality(G, nodi_fonte)
#since degree_centrality is a dict I can use that to get hubs
hub_degree = sorted(degree_centrality.items(),key= lambda x:x[1], reverse= True)[0]
hub_close = sorted(closeness_centrality.items(),key= lambda x:x[1], reverse= True)[0]
hub_betw = sorted(betweenness_centrality.items(),key= lambda x:x[1], reverse= True)[0]

print('degree centrality:      ',hub_degree)
print('betweenness: ',hub_betw)
print('closeness:   ',hub_close)


# ## Projection on Users
# Analize links between Users
#If 2 Users cite the same Source then a link will be created between them
#from graph get all User nodes
user_nodes = []
for (p, d) in G.nodes(data=True):
    if d['bipartite'] == 1:
        user_nodes.append(p)
        
projected_USER = bipartite.weighted_projected_graph(G, user_nodes, ratio=False)
print(projected_USER.number_of_nodes())
print(projected_USER.number_of_edges())
projected_USER_ref = get_list(projected_USER)

print(projected_USER_ref.number_of_nodes())
print(projected_USER_ref.number_of_edges())
print(projected_USER.number_of_nodes())
#save without islands
nx.write_gexf(projected_USER_ref,'projected_USER_ref.gexf')


# ### Community Detection on Refined Graph
# #### Modularity Optimization
# **Modularity** measures the density of connections within clusters compared to the density of connections between clusters (Blondel 2008). It is used as an objective function to be maximized for some community detection techniques and takes on values between -1 and 1. Graphs with a high modularity score will have many connections within a community but only few pointing outwards to other communities.

import networkx.algorithms.community as nx_comm
list_community_sets_greedy = list(nx_comm.greedy_modularity_communities(projected_USER_ref))
#get number of partitions
len(list(nx_comm.greedy_modularity_communities(projected_USER_ref)))

# #### Louvain 
# The Louvain Community Detection method is a simple algorithm that can quickly find clusters with high modularity in large networks.

partition_louvain = community_louvain.best_partition(projected_USER_ref)
len(set(partition_louvain.values()))

# #### Infomap
# The algorithm repeats the two described phases until an objective function is optimized. However, as an objective function to be optimized, Infomap does not use modularity but the so-called map equation

def findCommunities_infomap(im):

    print("Find communities with Infomap...")
    im.run();
    
    print(f"Found {im.num_top_modules} modules with codelength: {im.codelength}")
    communities = {}
    for node in im.tree:
        if node.is_leaf:
            #print(node.node_id, node.module_id)
            communities[node.node_id] = node.module_id
    
    return communities

im = Infomap()
G_num = nx.convert_node_labels_to_integers(projected_USER_ref, first_label=0)
pairs = [e for e in G_num.edges()]

im.add_links( pairs )

partition_infomap = findCommunities_infomap(im = im)


# #### Label Propagation
# Find communites through propagation process within the graph

partition_label = dict()
partition_label = nx.community.label_propagation_communities(projected_USER_ref)
print(type(partition_label))
communities = list(nx.community.label_propagation_communities(projected_USER_ref))
print(type(communities))
#print(communities)
print('number of communities: ', len(communities))


# ### Community Quality

from collections import defaultdict
list_community_sets_lpa = list(nx_comm.label_propagation_communities(projected_USER_ref))
list_community_sets_infomap = defaultdict(set)
for n, comm in partition_infomap.items():
    list_community_sets_infomap[comm].add(n)
    
list_community_sets_louvain= defaultdict(set)
for n, comm in partition_louvain.items():
    list_community_sets_louvain[comm].add(n)
    
quality =[[nx_comm.coverage(projected_USER_ref, my_list),nx_comm.modularity(projected_USER_ref, my_list, weight='weight'),nx_comm.performance(projected_USER_ref, my_list) ] for my_list in [list_community_sets_greedy,  list_community_sets_louvain.values(), list_community_sets_lpa ] ]

method_names = ["Greedy","Louvain library","LPA"]

plt.figure(figsize = (10,5))
sns.heatmap(pd.DataFrame(quality),annot = True, cmap = 'coolwarm', fmt = '.3')
plt.xticks([x+0.5 for x in range(3)],['Coverage','Modularity','Performance'], rotation = 0)
plt.yticks([x+0.5 for x in range(3)],method_names, rotation = 0)
plt.title('Community detection algorithm quality\n', weight = 'bold')
plt.show()


# ### Community size Distribution

comm_len = [(c, len(nodes)) for c,nodes in list_community_sets_louvain.items()]
comm_len.sort(key = lambda x: x[1],reverse = True)
plt.figure(figsize = (20,16))
plt.bar(range(len(comm_len)), [x[1] for x in comm_len],width = 1)
plt.xticks(range(len(comm_len)), [x[0] for x in comm_len])

plt.show()


# #### K-clique communities
# A k-clique community is the union of all cliques of size k that can be reached through adjacent (sharing k-1 nodes) k-cliques.

from networkx.algorithms.community import k_clique_communities
k_communities = k_clique_communities(projected_USER_ref, 15)
print(len(list(k_communities)))
#if 2 sources are cited by the same user then they are linked

#from graph get all Source nodes
source_nodes = []
for (p, d) in G.nodes(data=True):
    if d['bipartite'] == 0:
        source_nodes.append(p)
        
projected_SOURCE = bipartite.weighted_projected_graph(G, source_nodes, ratio=False)
print(projected_SOURCE.number_of_nodes())
print(projected_SOURCE.number_of_edges())

nx.write_gexf(projected_SOURCE,'projected_SOURCE.gexf')
projected_SOURCE_ref = get_list(projected_SOURCE)

print(projected_SOURCE_ref.number_of_nodes())
print(projected_SOURCE_ref.number_of_edges())
largest_cc = max(nx.connected_components(projected_SOURCE_ref), key=len)
len(set(largest_cc))/projected_SOURCE_ref.number_of_nodes()
nx.write_gexf(projected_SOURCE_ref,'projected_SOURCE_ref.gexf')



