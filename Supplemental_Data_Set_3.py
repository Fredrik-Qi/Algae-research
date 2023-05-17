# code_#6 表型层次聚类

#代码段1：用于输出图像
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

sns.set_theme()

# Load the data from the excel file
file_path = ""  #表型数据
sheet_names = pd.ExcelFile(file_path).sheet_names

for sheet_name in sheet_names:
    # Load data from sheet
    df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)

    # Normalize the data
    df_norm = df - 1
    #print(df_norm)
    
    # Draw the clustermap
    g = sns.clustermap(df_norm, cmap='RdBu_r', center=0, figsize=(120, 120), vmin=-1, vmax=1)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=8, rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=8)

    # Get the reordered data from the clustermap
    reordered_data = g.data2d
    
    # Save the reordered data to a new Excel file with sheet name as filename
    writer = pd.ExcelWriter(f"\\cluster_results_{sheet_name}.xlsx")  #输出按照层次聚类后重排序的excel文件
    reordered_data.to_excel(writer, index_label='Row', sheet_name='Reordered Data')
    writer.save()
    
    # Save the plot with sheet name as filename
    #g.savefig(f"\\cluster_heatmap_{sheet_name}.png", dpi=300)  #输出PNG图像
    # Save the plot with sheet name as filename in PDF format
    g.savefig(f"\\cluster_heatmap_{sheet_name}.pdf", format='pdf', dpi=300)  #输出PDF图像


#代码段2：用于输出聚类EXCEL
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Load the data from the Excel file
file_path = ""  #表型数据
sheet_names = pd.ExcelFile(file_path).sheet_names

# Set parameters for clustering
n_clusters = 10
linkage = "average"

# Loop over all sheets in the Excel file
for sheet_name in sheet_names:
    # Load data from sheet
    df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)

    # Normalize the data
    df_norm = pd.DataFrame(zscore(df), index=df.index, columns=df.columns)

    # Perform hierarchical clustering
    cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    cluster_labels = cluster.fit_predict(df_norm)

    # Add the cluster labels to the normalized data
    df_norm['cluster_label'] = cluster_labels

    # Create a dictionary to store the data frames for each cluster
    cluster_dict = {}
    for i in range(n_clusters):
        cluster_dict[i] = df_norm[df_norm['cluster_label'] == i].drop(columns=['cluster_label'])

    # Export the data frames to Excel
    writer = pd.ExcelWriter(f"\\clustered_data_{sheet_name}.xlsx")  #输出表型数据的聚类结果
    for i in range(n_clusters):
        cluster_dict[i].to_excel(writer, sheet_name=f"Cluster {i}")
    writer.save()

# code_#7 


