from kmeans import k_mean_method_and_display
import pandas as pd

# dataset for 1953
df1 = pd.read_csv('data1953.csv').to_numpy()
cluster_array_1953 = pd.array(df1)

# dataset for 2008
df2 = pd.read_csv('data2008.csv').to_numpy()
cluster_array_2008 = pd.array(df2)

# dataset for 1953 & 2008
df3 = pd.read_csv('dataBoth.csv').to_numpy()
cluster_array_both = pd.array(df3)

# running k-means method, uncomment to run a scenario
#k_mean_method_and_display(1000, 3, cluster_array_1953)
#k_mean_method_and_display(1000, 3, cluster_array_2008)
k_mean_method_and_display(1000, 4, cluster_array_both)
