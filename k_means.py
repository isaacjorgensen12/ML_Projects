#Isaac Jorgensen
#4/16/2019
#K-means employed to identify flowers in the Iris dataset

import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt


data = pd.read_excel('Iris.xls','Sheet1')

#remove the actual outcomes and replace with a column to keep the data point's current cluster assignment
actual_outcome = data['outcome(Cluster Index)']
working_data = data.drop(['Sample Index', 'outcome(Cluster Index)'], axis=1)
#working_data['working outcome(Cluster Index)'] = ""
working_data  = working_data.to_numpy()


#randomly choose k=3 centers
center1, center2, center3 = rd.sample(range(0, len(data.index)-1), 3)
centers = [working_data[center1], working_data[center2], working_data[center3]]


#
j_tracker = [0,0]
r = np.zeros((150,3))
iter = 0

#calculates the distance between the datapoint and each of the 3 centers, returns the index of the closest center
def minDist(datapoint):
    dist = [0, 0, 0]
    for k, c in enumerate(centers):
        dist[k] = np.linalg.norm(centers[k] - datapoint)
    return dist.index(min(dist))


def J(iteration):
    j = 0
    for n, d in enumerate(working_data):
        for k, c in enumerate(centers):
            j += r[n,k] * np.linalg.norm(c - d)
    return j



epsilon = 0.00010
#loop while 𝐽(Iter − 1) − 𝐽(Iter) < ε, where ε = 10^−5
while (j_tracker[iter-1] - j_tracker[iter]) < epsilon:
    
    #for each element in the set, calculate which center it's closest to and mark '1' at
    #its corresponding position in the r matrix
    for index, elmnt in enumerate(working_data):
        r[index, minDist(elmnt)] = 1

    #calculate the value of the objective function for this iteration and store it
    j_tracker.append(J(iter))
    iter += 1

    #take avg of all points associated with each center, set as new center
    for kindex in enumerate(centers):
        members = 0
        c0, c1, c2, c3 = 0, 0, 0, 0
        for nindex in enumerate(working_data):
            #sum all of the values for each column for each member of the subset
            c0 += r[nindex[0], kindex[0]] * working_data[nindex[0], 0]
            c1 += r[nindex[0], kindex[0]] * working_data[nindex[0], 1]
            c2 += r[nindex[0], kindex[0]] * working_data[nindex[0], 2]
            c3 += r[nindex[0], kindex[0]] * working_data[nindex[0], 3]
            #keep track of the total number of members in the current subset
            if r[nindex[0], kindex[0]] == 1: members += 1
        #before moving to the next centroid, average the value for each column/feature to
        #determing the new centroid for that subset
        centers[kindex[0]] = [c0/members, c1/members, c2/members, c3/members]
#end loop

#print(j_tracker[2:])
plt.plot(range(0,iter), j_tracker[2:])
plt.show()
