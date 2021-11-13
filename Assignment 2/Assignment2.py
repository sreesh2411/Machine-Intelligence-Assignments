def Extract(lst,n): 
    return [item[n] for item in lst]


def A_star_Traversal(cost, heuristic, start_point, goals):
    #v = [s,a,b(11),d,c(12),e,f]
    #S = []
    #pq = []
    l=[]
    pq=[]
    visited=[]
    number_of_nodes=len(cost[0])
    if start_point in goals:
        return [start_point]
    visited.append([heuristic[start_point], start_point, heuristic[start_point], -1]) #(a* score,node,heuristic,parent_node)
    pq.append([heuristic[start_point] ,start_point, heuristic[start_point], -1])
    successors=[]
    parent_child=[]
    path=[]
    flag = 0
    while(len(pq)!=0):
        q = pq.pop()
        while (q[1] in Extract(visited,1) and q[1]!=start_point):
            if(len(pq)!=0):
                q = pq.pop()
            else:
                flag = 1
                break
        if(flag==1):
            break
        if (q[1]!=start_point and q[1] not in Extract(visited,1)):
            #visited.append([heuristic[q[1]] + cost[q[3]][q[1]] , q[1] , heuristic[q[1]], q[3]])
            visited.append(q)
        for i in range(1,number_of_nodes):
            if(cost[q[1]][i]>0 and i not in Extract(visited,1)):
                successors.append(i)
                parent_child.append([q[1],i])
            if (cost[q[1]][i]>0 and i in Extract(visited,1)):
                f = heuristic[i] + cost[q[1]][i] + q[0] - q[2]
                for elem in visited:
                    if (i==elem[1] and f < elem[0]):
                        visited.append([f,i,heuristic[i],q[1]])
                        visited.remove(elem)
                        successors.append(i)
                        parent_child.append([q[1],i]) # [[s,a], [s,b], [s, d],]
        #print(successors)
        for i in successors:
            if (i not in goals):
                f = heuristic[i] + cost[q[1]][i] + q[0] - q[2]
                pq.append([f, i,heuristic[i], q[1]]) # 2 Bs added here
            else:
                #print("success")
                f = heuristic[i] + cost[q[1]][i] + q[0] - q[2]
                child = i
                temp=[]#[g2,c,d,s]
                temp.append(child) 
                temp2=[]#[s]
                while(child!=start_point):
                    for elem in parent_child:
                        if (child==elem[1]):
                            temp2.append(elem[0])
                            
                    temp4=[]
                    for k in visited:
                        for l in temp2:
                            if(l==k[1]):
                                temp4.append([k[0]-k[2]+cost[l][child],l])
                    temp4.sort()
                    temp.append(temp4[0][1])
                    child=temp4[0][1]
                    
                    temp2.clear()
                #temp.append(start_point)
                temp.reverse()
                path.append([f,temp]) #[[4,[1,2,6]],]
                #print(path)                    
                
        pq.sort(reverse=True)
        successors.clear()
        if (i[1] not in Extract(visited,1) for i in pq):
            if (i[0]<x for x in Extract(path,0)):
                pass
            else:
                break
            
    path.sort()
    l=path[0][1]
    #print("a*",path)
    return l
 

def UCS_Traversal(cost, start_point, goals
    #add your parameters 
):
    l = []
    pq=[]
    visited=[]
    if start_point in goals:
        return [start_point]
    visited.append([0, start_point, -1]) #(total_cost,node,parent_node)
    pq.append([0, start_point, -1])
    successors=[]
    parent_child=[]
    path=[]
    flag = 0
    number_of_nodes=len(cost[0])
    while(len(pq)!=0):
        q = pq.pop()
        while (q[1] in Extract(visited,1) and q[1]!=start_point):
            if(len(pq)!=0):
                q = pq.pop()
            else:
                flag = 1
                break
        if(flag==1):
            break                
        if (q[1]!=start_point and q[1] not in Extract(visited,1)):
            #visited.append([heuristic[q[1]] + cost[q[3]][q[1]] , q[1] , heuristic[q[1]], q[3]])
            visited.append(q)
        for i in range(1,number_of_nodes):
            if(cost[q[1]][i]>0 and i not in Extract(visited,1)):
                successors.append(i)
                parent_child.append([q[1],i])
            if (cost[q[1]][i]>0 and i in Extract(visited,1)):
                f = cost[q[1]][i] + q[0]
                for elem in visited:
                    if (i==elem[1] and f < elem[0]):
                        visited.append([f,i,q[1]])
                        visited.remove(elem)
                        successors.append(i)
                        parent_child.append([q[1],i]) # [[s,a], [s,b], [s, d],]
        #print(successors)
        for i in successors:
            if (i not in goals):
                f = cost[q[1]][i] + q[0]
                pq.append([f, i, q[1]]) # 2 Bs added here
            else:
                #print("success")
                f = cost[q[1]][i] + q[0]
                child = i
                temp=[]#[g2,c,d,s]
                temp.append(child) 
                temp2=[]#[s]
                while(child!=start_point):
                    for elem in parent_child:
                        if (child==elem[1]):
                            temp2.append(elem[0])
                    temp4=[]
                    for k in visited:
                        for l in temp2:
                            if(l==k[1]):
                                temp4.append([k[0]+cost[l][child],l])
                    temp4.sort()
                    if(len(temp4)==1):
                        temp.append(temp4[0][1])
                        child=temp4[0][1]
                    elif(len(temp4)>1 and temp4[0][0]!=temp4[1][0]):
                        temp.append(temp4[0][1])
                        child=temp4[0][1]
                    elif(len(temp4)>1 and temp4[0][0]==temp4[1][0]):
                        xyz = temp4[0][1]
                        for a in range(1,len(temp4)):
                            while(temp4[0][0]==temp4[a][0]):
                                xyz = temp4[a][1]
                                break
                        temp.append(xyz)
                        child=xyz
                        
                    temp2.clear()
                #temp.append(start_point)
                temp.reverse()
                path.append([f,temp]) #[[4,[1,2,6]],]
                #print(path)                    
                
        pq.sort(reverse=True)
        #print("pq ",pq)
        #print("visited ",visited)
        successors.clear()
    path.sort()
    l=path[0][1]
    #print("ucs",path)
    return l


def DFS_Traversal(cost, start_point, goals):
    l = []
    pq={}
    visited=[]
    length=len(cost[0])-1
    item=0
    val=[0 for i in range(length+2)]
    visited.append((start_point,0))
    while(len(visited)!=0):
        current=visited.pop()
        if(val[current[0]]==0):
            pq[current[0]]=current[1]
            if current[0] in goals:
                item=current[0]
                break
            val[current[0]]=1


        for i in range(length,0,-1):
            if(i!=current[0] and val[i]==0 and cost[current[0]][i]!=-1):
                visited.append((i,current[0]))

    if(item==0):
        return []
    else:
        check=item
        while(check!=start_point):
            l.append(pq[check])
            check=pq[check]
        l=l[::-1]
        l.append(item)
    return l


'''
Function tri_traversal - performs DFS, UCS and A* traversals and returns the path for each of these traversals 

n - Number of nodes in the graph
m - Number of goals ( Can be more than 1)
1<=m<=n
Cost - Cost matrix for the graph of size (n+1)x(n+1)
IMP : The 0th row and 0th column is not considered as the starting index is from 1 and not 0. 
Refer the sample test case to understand this better

Heuristic - Heuristic list for the graph of size 'n+1' 
IMP : Ignore 0th index as nodes start from index value of 1
Refer the sample test case to understand this better

start_point - single start node
goals - list of size 'm' containing 'm' goals to reach from start_point

Return : A list containing a list of all traversals [[],[],[]]
1<=m<=n
cost[n][n] , heuristic[n][n], start_point, goals[m]

NOTE : you are allowed to write other helper functions that you can call in the given fucntion
'''

def tri_traversal(cost, heuristic, start_point, goals):
    l = []

    t1 = DFS_Traversal(cost, start_point, goals
    #send whatever parameters you require 
)
    t2 = UCS_Traversal(cost, start_point, goals
    #send whatever parameters you require 
)
    t3 = A_star_Traversal(cost, heuristic, start_point, goals
    #send whatever parameters you require 
)

    l.append(t1)
    l.append(t2)
    l.append(t3)
    #print(l)
    return l

