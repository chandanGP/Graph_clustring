import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time

def import_facebook_data(file_path):
    print("Reading facebook data ",end='')
    output=set()
    with open(file_path, 'r') as file:
        for line in file:
            temp=line[:-1].split(' ')
            output.add((int(temp[0]),int(temp[1])))
    output=np.array(list(output))
    print("....Done\n")
    return np.unique(output[np.argsort(output[:,0])],axis=0)

def import_bitcoin_data(file_path):
    print("Reading bitcoin data ",end='')
    df = pd.read_csv(file_path,header=None)
    data=df.values[:,:-2].astype('int')
    print("....Done\n")
    return np.unique(data[np.argsort(data[:,0])],axis=0)

def spectralDecomp_OneIter(nodes_connectivity_list,plot=True):
    if(plot):
        print("Spectral decomposition based clustring one iteration")

    st=time.time()
    G=nx.Graph()
    G.add_edges_from(np.vstack((nodes_connectivity_list,np.transpose(np.vstack((nodes_connectivity_list[:,1],nodes_connectivity_list[:,0]))))))
    Adj_mat=nx.to_numpy_array(G)

    diag_vect=Adj_mat.sum(axis=1)
    L=np.diag(diag_vect)-Adj_mat
    #min cut
    eigen_values, eigen_vectors = np.linalg.eig(L)

    temp=np.unique(eigen_values)
    temp.sort()
    
    fiedler_vector=None
    clus_vect_mincut=None
    for min_eigen_val in temp:
        if(min_eigen_val>0):
            fiedler_vector=np.real(np.transpose(eigen_vectors[:,eigen_values==min_eigen_val])[0])
            clus_vect_mincut=(fiedler_vector>=0)
            temp_sum=clus_vect_mincut.sum()
            if(temp_sum>0 and temp_sum<clus_vect_mincut.shape[0]):
                break

    #print(temp[1])
    #fiedler_vector=np.real(np.transpose(eigen_vectors[:,eigen_values==min_eigen_val])[0])
    sorted_fiedler_vector_idx=np.argsort(fiedler_vector)
    sorted_fiedler_vector=fiedler_vector[sorted_fiedler_vector_idx]
    #clus_vect_mincut=(fiedler_vector>=0)
    node_color=['green' if i else 'red' for i in clus_vect_mincut]


    clusters=np.zeros(clus_vect_mincut.shape[0],dtype='int')

    graph_partition=[]
    nodeId_list=np.array(G.nodes())
    minNode_id_c0=np.min(nodeId_list[clus_vect_mincut==False])
    minNode_id_c1=np.min(nodeId_list[clus_vect_mincut==True])
    clusters[clus_vect_mincut==True]=minNode_id_c1
    clusters[clus_vect_mincut==False]=minNode_id_c0
    graph_partition=np.transpose(np.vstack((nodeId_list,clusters)))
    num_nodes_c1=clus_vect_mincut.sum()
    num_nodes_c0=clus_vect_mincut.shape[0]-num_nodes_c1
    et=time.time()

    if(plot):
        #sorted fiedler vector
        print("ploting Sorted fielder vector")
        plt.figure(figsize=(10,10))
        plt.plot(range(len(sorted_fiedler_vector)),sorted_fiedler_vector,linestyle='dotted')
        plt.title("Sorted fielder vector")
        plt.xlabel("Sorted Nodes")
        plt.ylabel("Fielder vector values")
        plt.show()

        #Adjacency matirx
        print("ploting Adjacency matrix before sorting nodes")
        plt.figure(figsize=(10,10))
        plt.imshow(Adj_mat, cmap='hot', interpolation='none')
        plt.title("Adjacency matrix before sorting nodes")
        plt.xlabel("Nodes")
        plt.ylabel("Nodes")
        plt.show()

        #sorted adjacency matrix
        Adj_mat=createSortedAdjMat(graph_partition, nodes_connectivity_list,plot=False)
        print("ploting Adjacency matrix after sorting nodes")
        plt.figure(figsize=(10,10))
        plt.imshow(Adj_mat, cmap='hot', interpolation='none')
        plt.title("Adjacency matrix after sorting nodes")
        plt.xlabel("Nodes")
        plt.ylabel("Nodes")
        plt.show()

        #visualising graph
        print("visualising graph")
        plt.figure(figsize=(10,10))
        nx.draw_networkx(G,with_labels=False,node_size=15,node_color=node_color,width=0.4)
        plt.title("clustered graph")
        plt.show()

        print("Cluster ID:",minNode_id_c0,"  number of nodes:",num_nodes_c0)
        print("Cluster ID:",minNode_id_c1,"  number of nodes:",num_nodes_c1)
        print("Time taken for clustring:",et-st," (s)\n")
    
    return fiedler_vector, Adj_mat, graph_partition

def spectralDecomposition(nodes_connectivity_list):
    print("Spectral decomposition multiple iterations ....",end='')
    st=time.time()
    G=nx.Graph()
    G.add_edges_from(np.vstack((nodes_connectivity_list,np.transpose(np.vstack((nodes_connectivity_list[:,1],nodes_connectivity_list[:,0]))))))
    Adj_mat=nx.to_numpy_array(G)
    m2=Adj_mat.sum()
    
    node_neighbours={}
    nodewise_edges={}
    nodes=np.array(G.nodes)
    for node in nodes:
        temp_edges_idxs0=np.where(nodes_connectivity_list[:,0]==node)
        temp_edges_idxs1=np.where(nodes_connectivity_list[:,1]==node)
        nodewise_edges[node]=nodes_connectivity_list[temp_edges_idxs0]
        node_neighbours[node]=np.append(nodes_connectivity_list[temp_edges_idxs0,1],nodes_connectivity_list[temp_edges_idxs1,0])

    def Q(clus_nodes):
        sigma_total=0
        sigma_in=0
        for node in clus_nodes:
            sigma_total+=node_neighbours[node].shape[0]
            sigma_in+=len(np.intersect1d(node_neighbours[node],clus_nodes))
        return (sigma_in/m2)-((sigma_total/m2)**2)
    
    min_node_id=np.min(nodes)
    clusters={min_node_id:{'Q':Q(nodes),'nodes':nodes}}

    splitted=True
    itr=0
    after_Q=0
    while splitted:
        itr+=1
        #print("iteration",itr,":")
        temp_dict={}
        splitted=False
        after_Q=0
        for cluster in clusters.keys():
            #print("cluster:",cluster)
            PQ=clusters[cluster]['Q']
            #print("PQ:",PQ)
            Pnodes=clusters[cluster]['nodes']
            nodes_connect=[]
            for node in Pnodes:
                for edge in nodewise_edges[node]:
                    if(edge[1] in Pnodes):
                        nodes_connect.append(edge)
            nodes_connect=np.array(nodes_connect)
            nodes_connect=np.unique(nodes_connect,axis=0)

            if(nodes_connect.shape[0]>0):
                _, _, graph_partition = spectralDecomp_OneIter(nodes_connect,plot=False)
                cls_ids=np.unique(graph_partition[:,1])
                cls0=graph_partition[:,0][np.where(graph_partition[:,1]==cls_ids[0])]
                set_cls0=set(cls0)
                cls1=graph_partition[:,0][np.where(graph_partition[:,1]==cls_ids[1])]
                set_cls1=set(cls1)
                cls1=np.array([],dtype='int')
                cls0=np.array([],dtype='int')
                for node in Pnodes:
                    if(node in set_cls1 ):
                        cls1=np.append(cls1,node)
                    else:
                        cls0=np.append(cls0,node)
                CQ0=Q(cls0)
                CQ1=Q(cls1)
                #print("num nodes before:",len(Pnodes)," num nodes after:",len(graph_partition))
                #print("CQ0:",CQ0)
                #print("CQ1:",CQ1)
                if(CQ0+CQ1 > PQ):
                    #print("splitting")
                    splitted=True
                    temp_dict[np.min(cls0)]={'Q':CQ0,'nodes':cls0}
                    temp_dict[np.min(cls1)]={'Q':CQ1,'nodes':cls1} 
                else:
                    #print("not splitting by Q val")
                    temp_dict[cluster]={'Q':PQ,'nodes':Pnodes}
            else:
                #print("not splitting  by dimention:",len(Pnodes))
                temp_dict[cluster]={'Q':PQ,'nodes':Pnodes}
        clusters=temp_dict
        #print("before Q:",before_Q)
        after_Q=0
        for cls in clusters.keys():
            after_Q+=Q(clusters[cls]['nodes'])
        #print("after_Q:",after_Q)
        #print("number of clusters:",len(clusters))
        #print("\n")

    graph_partition=[]
    for clus in clusters.keys():
        for node in clusters[clus]['nodes']:
            graph_partition.append([node,clus])
    graph_partition=np.array(graph_partition,dtype='int')
    et=time.time()

    print("Done")
    for clus in clusters.keys():
        print("Cluster ID:",clus,"  number of nodes:",clusters[clus]['nodes'].shape[0])
    print("Total number of clusters:",len(clusters))
    print("Final modularity:",after_Q)
    print("Time taken for clustring:",et-st," (s)\n")
    
    return graph_partition

def createSortedAdjMat(graph_partition, nodes_connectivity_list,plot=True):

    G=nx.Graph()
    G.add_edges_from(np.vstack((nodes_connectivity_list,np.transpose(np.vstack((nodes_connectivity_list[:,1],nodes_connectivity_list[:,0]))))))
    nodeId_clus_map={}
    for node_id, clus in graph_partition:
        nodeId_clus_map[node_id]=clus

    
    clusters=np.sort(np.unique(graph_partition[:,1]))
    sorted_nodes=np.array([],dtype='int')
    for clus in clusters:
        sorted_nodes=np.append(sorted_nodes,np.sort(graph_partition[graph_partition[:,1]==clus][:,0]))
    
    nodeId_idx_map={}
    for idx, node_id in enumerate(sorted_nodes):
        nodeId_idx_map[node_id]=idx
    
    dim=graph_partition.shape[0]
    Adj_mat=np.zeros((dim,dim))
    for node1, node2 in nodes_connectivity_list:
        Adj_mat[nodeId_idx_map[node1]][nodeId_idx_map[node2]]=1
        Adj_mat[nodeId_idx_map[node2]][nodeId_idx_map[node1]]=1
    
    if(plot):
        print("ploting adjacency matrix")
        plt.figure(figsize=(10,10))
        plt.imshow(Adj_mat, cmap='hot', interpolation='none')
        plt.title("Adjacency matrix nodes sorted according to clusters")
        plt.xlabel("Nodes")
        plt.ylabel("Nodes")
        plt.show()

        cluster_colors={}
        for clus, clr in zip(clusters,plt.get_cmap('viridis',len(clusters))(np.arange(len(clusters)))):
            cluster_colors[clus]=clr
        
        node_colors={}
        for node_id, clus in graph_partition:
            node_colors[node_id]=cluster_colors[clus]
        
        node_color=[]
        for node_id in G.nodes():
            node_color.append(node_colors.get(node_id,'red'))
        print("visualising graph")
        layout = nx.spring_layout(G)
        plt.figure(figsize=(10,10))
        nx.draw_networkx(G,with_labels=False,pos=layout,node_size=15,node_color=node_color,width=0.4)
        plt.title("clustered graph")
        plt.show()
        print("\n")
        
    return Adj_mat

def louvain_one_iter(nodes_connectivity_list):
    print("Running one iteration of Louvain algorithm ....",end='')
    st=time.time()
    G=nx.Graph()
    G.add_edges_from(np.vstack((nodes_connectivity_list,np.transpose(np.vstack((nodes_connectivity_list[:,1],nodes_connectivity_list[:,0]))))))
    Adj_mat=nx.to_numpy_array(G)
    
    Adj_mat=Adj_mat/Adj_mat.sum()
    #initialising
    deg_vect=np.sum(Adj_mat,axis=1)

    temp=np.where(Adj_mat!=0)
    node_neighbours={}
    for node_idx in range(deg_vect.shape[0]):
        node_neighbours[node_idx]=set(temp[1][temp[0]==node_idx])

    node_to_cluster_map={i:i for i in range(deg_vect.shape[0])}
    cluster_to_node_map={i:{'sigma_in':Adj_mat[i][i],'sigma_clus':deg_vect[i],'nodes':{i}} for i in range(deg_vect.shape[0])}

    #louvian algo
    changed=True
    current_modularity=-1
    count=-1
    while changed :
        count+=1
        #print("iteration ",count,":")
        #changed=False
        num_changed=0
        for node_idx in range(deg_vect.shape[0]):
            max_delta_Q=0
            from_clus=node_to_cluster_map[node_idx]
            to_clus=node_to_cluster_map[node_idx]
            max_sigma_in_oc_after=None
            max_sigma_oc_after=None
            max_sigma_in_nc_after=None
            max_sigma_nc_after=None


            old_clus_idx=node_to_cluster_map[node_idx]
            Kio=2*Adj_mat[node_idx][list(cluster_to_node_map[old_clus_idx]['nodes'])].sum()
            sigma_in_oc_after=cluster_to_node_map[old_clus_idx]['sigma_in']-Kio
            sigma_oc_after=cluster_to_node_map[old_clus_idx]['sigma_clus']-deg_vect[node_idx]
            delta_Q_demerge=(2*deg_vect[node_idx]*(cluster_to_node_map[old_clus_idx]['sigma_clus']-deg_vect[node_idx]))-Kio
            
            visited={}
            for neighbour_idx in node_neighbours[node_idx]:
                new_clus_idx=node_to_cluster_map[neighbour_idx]
                if(visited.get(new_clus_idx,True)):
                    visited[new_clus_idx]=False
                    Kin=2*Adj_mat[node_idx][list(cluster_to_node_map[new_clus_idx]['nodes'])].sum()
                    sigma_in_nc_after=cluster_to_node_map[new_clus_idx]['sigma_in']+Kin
                    sigma_nc_after=cluster_to_node_map[new_clus_idx]['sigma_clus']+deg_vect[node_idx]
                    delta_Q_merge=Kin-(2*cluster_to_node_map[new_clus_idx]['sigma_clus']*deg_vect[node_idx])
                    if(delta_Q_demerge+delta_Q_merge > max_delta_Q):
                        #print("updating... for:",node_idx)
                        max_delta_Q=delta_Q_demerge+delta_Q_merge
                        #print("max_delta_Q:",max_delta_Q)
                        to_clus=new_clus_idx
                        max_sigma_in_oc_after=sigma_in_oc_after
                        max_sigma_oc_after=sigma_oc_after
                        max_sigma_in_nc_after=sigma_in_nc_after
                        max_sigma_nc_after=sigma_nc_after
            if(from_clus!=to_clus):
                #print("node:",node_idx," changed from:",from_clus," to:",to_clus)
                num_changed+=1
                #return
                #make changes accordnig to optimal change in modularity
                #remove node from it's current cluster
                cluster_to_node_map[from_clus]['nodes'].remove(node_idx)
                cluster_to_node_map[from_clus]['sigma_in']=max_sigma_in_oc_after
                cluster_to_node_map[from_clus]['sigma_clus']=max_sigma_oc_after
                #add node to new cluster
                cluster_to_node_map[to_clus]['nodes'].add(node_idx)
                cluster_to_node_map[to_clus]['sigma_in']=max_sigma_in_nc_after
                cluster_to_node_map[to_clus]['sigma_clus']=max_sigma_nc_after
                
                node_to_cluster_map[node_idx]=to_clus
    
        #termination condition
        #computing total  modularity 
        new_modularity=0
        for node_idx in range(deg_vect.shape[0]):
            same_cluster_nodes_idx=list(cluster_to_node_map[node_to_cluster_map[node_idx]]['nodes'])
            new_modularity+=(Adj_mat[node_idx][same_cluster_nodes_idx] - (deg_vect[node_idx]*deg_vect[same_cluster_nodes_idx])).sum()
        #print("iteration:",count," new_modularity:",new_modularity,' dist modularity:',abs(new_modularity-1),' num_nodes_changed:',num_changed,"\n")
        #break
        if(num_changed==0):
            break
    node_ids=np.array(G.nodes()).astype('int')

    node_clusters=[]
    for clus in cluster_to_node_map.keys():
        temp=set()
        for node in cluster_to_node_map[clus]['nodes']:
            temp.add(node_ids[node])
        if(len(temp)>0):
            clus_id=min(temp)
            for node_id in temp:
                node_clusters.append([node_id,clus_id])
    graph_partition=np.array(node_clusters,dtype='int')
    et=time.time()
    print("Done")
    print("Number of clusters created:",len(np.unique(graph_partition[:,1])))
    print("Final modularity:",new_modularity)
    print("Time taken:",et-st," (s)")
    print("Ploting sorted adjacency matrix and graph")
    createSortedAdjMat(graph_partition, nodes_connectivity_list)
    
    return graph_partition

if __name__ == "__main__":

    
    ############ Answer qn 1-4 for facebook data #################################################
    # Import facebook_combined.txt
    # nodes_connectivity_list is a nx2 numpy array, where every row 
    # is a edge connecting i<->j (entry in the first column is node i, 
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.
    nodes_connectivity_list_fb = import_facebook_data("../data/facebook_combined.txt")

    # This is for question no. 1
    # fielder_vec    : n-length numpy array. (n being number of nodes in the network)
    # adj_mat        : nxn adjacency matrix of the graph
    # graph_partition: graph_partitition is a nx2 numpy array where the first column consists of all
    #                  nodes in the network and the second column lists their community id (starting from 0)
    #                  Follow the convention that the community id is equal to the lowest nodeID in that community.
    print("Question 1:one iteration of spectral decomposition based clustring on facebook data")
    fielder_vec_fb, adj_mat_fb, graph_partition_fb = spectralDecomp_OneIter(nodes_connectivity_list_fb)

    # This is for question no. 2. Use the function 
    # written for question no.1 iteratetively within this function.
    # graph_partition is a nx2 numpy array, as before. It now contains all the community id's that you have
    # identified as part of question 2. The naming convention for the community id is as before.
    print("Question 2:multiple iterations of spectral decomposition based clustring on facebook data")
    graph_partition_fb = spectralDecomposition(nodes_connectivity_list_fb)

    # This is for question no. 3
    # Create the sorted adjacency matrix of the entire graph. You will need the identified communities from
    # question 3 (in the form of the nx2 numpy array graph_partition) and the nodes_connectivity_list. The
    # adjacency matrix is to be sorted in an increasing order of communitites.
    print("Question 3:ploting sorted adjacency matrix and graph from result of question 2 on facebook data")
    clustered_adj_mat_fb = createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb)

    # This is for question no. 4
    # run one iteration of louvain algorithm and return the resulting graph_partition. The description of
    # graph_partition vector is as before.
    print("Question 4:Running one iteration of louvain algorithm and ploting adjacency matrix and graph on facebook data")
    graph_partition_louvain_fb = louvain_one_iter(nodes_connectivity_list_fb)


    ############ Answer qn 1-4 for bitcoin data #################################################
    # Import soc-sign-bitcoinotc.csv
    nodes_connectivity_list_btc = import_bitcoin_data("../data/soc-sign-bitcoinotc.csv")

    # Question 1
    print("Question 1:one iteration of spectral decomposition based clustring on bitcoin data")
    fielder_vec_btc, adj_mat_btc, graph_partition_btc = spectralDecomp_OneIter(nodes_connectivity_list_btc)

    # Question 2
    print("Question 2:multiple iterations of spectral decomposition based clustring on bitcoin data")
    graph_partition_btc = spectralDecomposition(nodes_connectivity_list_btc)

    # Question 3
    print("Question 3:ploting sorted adjacency matrix and graph from result of question 2 on bitcoin data")
    clustered_adj_mat_btc = createSortedAdjMat(graph_partition_btc, nodes_connectivity_list_btc)

    # Question 4
    print("Question 4:Running one iteration of louvain algorithm and ploting adjacency matrix and graph on bitcoin data")
    graph_partition_louvain_btc = louvain_one_iter(nodes_connectivity_list_btc)


