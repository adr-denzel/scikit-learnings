# reference: https://www.analyticsvidhya.com/blog/2021/04/k-means-clustering-simplified-in-python/
# reference: https://towardsdatascience.com/k-means-clustering-from-scratch-6a9d19cafc25
# reference: https://pandas.pydata.org/docs/reference/api/pandas.array.html
# reference: https://pandas.pydata.org/pandas-docs/version/0.24.0rc1/api/generated/pandas.DataFrame.to_numpy.html#pandas.DataFrame.to_numpy

# K-Means clustering implementation

"""
Process Overview Pseudocode

1) Data import, from csv to DataFrame, then use to_numpy() method to access data in frame.
2) Define function to calculate euclidean 2d distance given any 2 points.
3) Define function to establish a set of initial centroids at random, given the number of centroids sought.
4) Define function to assign each point in dataset to the index of it's nearest defined centroid in centroid list.
5) Define function to re-establish a set of centroids based on the x_mean, and y_mean of the points per current cluster.
6) Finally define the function for the major algorithms performing the k-mean clustering process:
    - Define initial set of centroids via function 2.
    - Assign all points to the index of the initial set of centroids via function 3.
    - Looping for x number of iterations the following is performed:
        * Replacement set of centroids is defined to via function 5 averaging the x_means and y_means.
        * Clusters are yet again re-established via calling function 4, based on the newly averaged centroids.
"""

# importing libs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rd

# data import
df1 = pd.read_csv('data2008.csv').to_numpy()
cluster_array = pd.array(df1)


# euclidean distance function
def calc_distance(p1, p2):
    return ((p1[1] - p2[0]) ** 2 + (p1[2] - p2[1]) ** 2) ** 0.5


# picking a random set of centroids to initialise the process with function
def initialise_centroids(num_of_centroids, data):
    # random centroid selection from input data
    init_centroid_indices = rd.sample(range(0, len(data)), num_of_centroids)
    first_centroids = []
    # appending centroids
    for i in init_centroid_indices:
        first_centroids.append(data[i])

    # removing country name data from centroids
    first_centroids_processed = []
    for i in first_centroids:
        first_centroids_processed.append(i[1:])

    return first_centroids_processed


# assigning the data to their closest centroid
def assign_closest_centroid(centroids_array, data):
    centroid_index_assignment_to_data = []
    # calculating the distance of each point to all their centroids
    for i in data:
        distances = []
        for j in centroids_array:
            distances.append(calc_distance(i, j))
            # record the index of the closest centroid
        centroid_index_assignment_to_data.append(np.argmin(distances))

    return centroid_index_assignment_to_data


# recalculate centroids based on means of assigned points to centroids
def recalc_centroids_based_on_mean(centroid_assignment, data):
    new_centroids = []
    # concat the index of the closest centroid per point to main DataTable
    merged_data = pd.concat([pd.DataFrame(data), pd.DataFrame(centroid_assignment,
                                                              columns=['cluster'])], axis=1).to_numpy()

    for i in set(centroid_assignment):
        cluster_to_average = []
        # append all points in the same cluster to list
        for j in merged_data:
            if i == j[3]:
                cluster_to_average.append(j[1:3])
        # turn cluster into DataFrame
        new_df = pd.DataFrame(cluster_to_average).to_numpy()
        # getting the means of the cluster
        new_centroids.append(new_df.mean(axis=0))

    return new_centroids


# process, outputs and visualisations
def k_mean_method_and_display(num_of_iterations, num_of_centroids, data):
    # initialise centroids
    centroids = initialise_centroids(num_of_centroids, data)
    # assign points to the clusters surrounding centroids
    clusters = assign_closest_centroid(centroids, data)

    # iterations of centroid definition by means, and reassigning the centroids defining the clusters
    for i in range(num_of_iterations):
        centroids = recalc_centroids_based_on_mean(clusters, data)
        clusters = assign_closest_centroid(centroids, data)

    # DataFrame for visuals
    new_df = pd.concat([pd.DataFrame(data, columns=['country', 'birthRate', 'lifeEx']),
                        pd.DataFrame(clusters, columns=['cluster'])], axis=1)

    # mean LEs and BRs for terminal
    print("\nCluster Means for Life Expectancy and Birth Rate")
    for i in range(len(centroids)):
        print("\nCluster {}".format(i + 1))
        print("Mean Birth Rate (live-births per 1000 people): {}".format(centroids[i][0]))
        print("Mean Life Expectancy (years): {}".format(centroids[i][1]))

    # clustered countries numbers
    print("\nNumber of Countries per Cluster")
    df_to_numpy = new_df.to_numpy()
    for i in range(len(centroids)):
        count = 0
        for j in df_to_numpy:
            if i == j[3]:
                count += 1
        print("\nCluster {}".format(i + 1))
        print("{} countries".format(count))

    # clustered countries list
    print("\nList of Countries per Cluster")
    df_to_numpy = new_df.to_numpy()
    for i in range(len(centroids)):
        country_list = ""
        for j in df_to_numpy:
            if i == j[3]:
                country_list += j[0]
                country_list += "\n"
        print("\nCluster {}".format(i + 1))
        print(country_list)

    # plt output
    plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], c='black')
    plt.scatter(new_df['birthRate'], new_df['lifeEx'], c=new_df['cluster'], cmap='rainbow')
    plt.show()


# testing it all
# k_mean_method_and_display(1000, 4, cluster_array)    # 1000 is the number of iterations, 4 is the number of centroids
