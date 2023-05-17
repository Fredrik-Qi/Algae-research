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

# code_#7 Mutual Best Hits

import os
#folder_path = ""  #输入数据文件夹
#Cre_db_path = "/Cre_blastdb"   #专门生成一个文件夹存放blastdb文件，且能避免重复构建Cre的blastdb


#1）BLAST建库
from Bio.Blast.Applications import NcbimakeblastdbCommandline
def create_blast_db(file_path,db_output_path):
    db_name = file_path.split("/")[-1].split(".")[0] + "_blastdb"
    db_path = os.path.join(db_output_path, db_name)
    cmd = NcbimakeblastdbCommandline(
        input_file=file_path,
        dbtype="prot",
        out= db_path ,
    )
    stdout, stderr = cmd()
    print(stdout)
    print(stderr)



#2）进行正向BLAST并输出结果
from Bio.Blast.Applications import NcbiblastpCommandline
def run_blastp(query_file, subject_file, file_path):
    out_file = f"{file_path}/a_to_b.blast"
    cmd = NcbiblastpCommandline(
        query=query_file,
        subject=subject_file,
        out=out_file,
        outfmt=6,
    )
    stdout, stderr = cmd()
    print(stdout)
    print(stderr)



#3）对正向BLAST的数据进行筛选，对于相同项的物种A基因，选择pident最高的物种B被比对基因
#或 对反向BLAST的数据进行筛选，对于相同项的物种B基因，选择pident最高的物种A被比对基因
import pandas as pd
def filter_blast_results(blast_output_path, output_path):
    # 读取BLAST结果
    blast_results = pd.read_csv(blast_output_path, sep='\t', header=None)

    # 筛选第三列值最小的结果
    filtered_results = blast_results.iloc[blast_results.groupby([0])[2].idxmax()]

    # 导出结果到文件
    filtered_results.to_csv(output_path, sep='\t', header=None, index=None)



#4) 基于第一次blast文件筛选后得到的物种B基因，在"物种B Protein FASTA"中筛选，得到新的fa文件
from Bio import SeqIO
#这个函数将会读取第一次blast文件，并从中提取出第二列的物种B基因名，然后读取"物种B Protein FASTA"文件
#对于每一个记录，如果它的ID在blast结果中出现过，就将它输出到输出文件中。
def filter_fasta(blast_file, fasta_file, output_file):
    # Read blast file and extract the second column
    with open(blast_file, 'r') as blast:
        blast_ids = set(line.strip().split()[1] for line in blast)

    # Read fasta file and filter by IDs
    with open(fasta_file, 'r') as fasta, open(output_file, 'w') as output:
        for record in SeqIO.parse(fasta, 'fasta'):
            if record.id in blast_ids:
                SeqIO.write(record, output, 'fasta')



#5）进行反向BLAST并输出结果
def run_blastp_1(query_file, subject_file, file_path):
    out_file = f"{file_path}/b_to_a.blast"
    cmd = NcbiblastpCommandline(
        query=query_file,
        subject=subject_file,
        out=out_file,
        outfmt=6,
    )
    stdout, stderr = cmd()
    print(stdout)
    print(stderr)




#6) 合并正向BLAST和反向BLAST的数据，选取mutual best hits，并将数据输出
import csv
def write_mutual_best_hit(b_to_a_path, a_to_b_path, output_file):
    b_to_a_dict = {}
    with open(b_to_a_path, 'r') as b_to_a_file:
        b_to_a_reader = csv.reader(b_to_a_file, delimiter='\t')
        for row in b_to_a_reader:
            b_to_a_dict[row[0]] = row[1:]

    with open(a_to_b_path, 'r') as a_to_b_file:
        with open(output_file, 'w', newline='') as output:
            writer = csv.writer(output, delimiter=',')
            a_to_b_reader = csv.reader(a_to_b_file, delimiter='\t')
            for row in a_to_b_reader:
                b_gene_name = row[1]
                if b_gene_name in b_to_a_dict and b_to_a_dict[b_gene_name][0] == row[0]:
                    writer.writerow(row + b_to_a_dict[b_gene_name][1:])


#第一步：单独为Cre建库
file1_path = f"{Cre_db_path}/Creinhardtii_281_v5.6.protein.fa"
create_blast_db(file1_path,Cre_db_path)
print("The library of species A was successfully established")



#第二步：执行后续分析程序
# 遍历指定目录下的所有文件夹
for dir_name in os.listdir(folder_path):
    file_path = folder_path + "/" + dir_name  # 拼接文件路径
    if dir_name == 'Cre_blastdb':  #跳过存储库的文件夹
        continue
    if os.path.isdir(file_path):   #判断其是否为文件夹
        print(file_path)
        #作为分析的核心程序，涵盖所有分析的函数
        def analyze_files():
            file2_path = f"{file_path}/species_B_genome_protein.fa"
            file3_path = f"{file_path}/sum_a.fa"

            create_blast_db(file2_path,file_path)
            print("The library of species B was successfully established")

            run_blastp(file3_path,file2_path,file_path)
            print("Forward blast has been completed")

            filter_blast_results(f"{file_path}/a_to_b.blast",f"{file_path}/a_to_b_filtered.blast")
            print("BLAST result filtering complete, best hits have been selected")

            filter_fasta(f"{file_path}/a_to_b_filtered.blast",file2_path,f"{file_path}/b.fa")
            print("Filtered fasta data of B was selected")

            run_blastp_1(f"{file_path}/b.fa",file1_path,file_path)
            print("Reverse blast has been completed")

            filter_blast_results(f"{file_path}/b_to_a.blast",f"{file_path}/b_to_a_filtered.blast")
            print("BLAST result filtering complete, best hits have been selected")

            write_mutual_best_hit(f"{file_path}/a_to_b_filtered.blast",f"{file_path}/b_to_a_filtered.blast",f"{file_path}/mutual_best_hit.csv")
            print("Mutual best hits complete")
        analyze_files()

        
# code_#8 Correlation of Evolution
#1）生成成对的相关系数
#2）基于相关系数的分布绘制柱状图
#3）绘制相关系数矩阵

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 读取excel文件
df = pd.read_excel('')

# 计算相关系数矩阵
corr_matrix = df.corr(method='pearson')

# 生成相关系数表格并保存到新的excel文件中
corr_matrix.to_excel('')

# 将相关系数矩阵转化为长格式数据
corr_matrix = corr_matrix.stack().reset_index()
corr_matrix.columns = ['variable_1', 'variable_2', 'correlation']
corr_matrix.to_excel('')


# 绘制柱状图
sns.histplot(data=corr_matrix, x='correlation', bins=200)
plt.xlabel('Correlation')
plt.ylabel('Count')
plt.title('Distribution of Correlations')
plt.show()


#筛选相关系数表格并保存到新文件中
df = corr_matrix
print(df)
df[(df > -0.8) & (df < 0.8)] = 0
df.to_excel('')


# code_#9 基因表达数据的相关系数矩阵（R语言）
install.packages("ggplot2")
install.packages("corrplot")
install.packages("readxl")
library(corrplot)
library(ggplot2)
library(readxl)
df <- read_excel("", sheet = "")
puredata <- apply(df, 2, as.numeric)  
#apply 函数是一种非常通用的数据处理函数，可以对数组、矩阵和列表等对象进行操作。
#第二个参数是指定处理数据的维度， 2表示按照列进行处理，1表示按行进行处理;该函数将导入的数据转化为数值型
cor_data <- cor(puredata)  #使用cor函数计算相关矩阵

#定义圈出的cluster，以及圈出线的颜色和线条
#diag = FALSE 剔除对角线的值；lwd为矩形框的线宽；addrect为矩形框数量
#tl.pos为标签的位置，默认'b'，标签在矩形框下。't'为上方，'d'为中间
#cor_data是之前计算得到的相关性矩阵，method指定了可视化的方式，这里选择了层次聚类；
corrplot(cor_data, method = 'square', diag = FALSE, order = 'hclust',   
         addrect = 5, 
         rect.col = 'black', 
         rect.lwd = 3, 
         tl.pos = 't',
         tl.cex=1)

# code_#10 基因表达数据的PCA主成分聚类分析
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 读取 Excel 文件
df = pd.read_excel("", sheet_name="")

# 获取数据
labels = df.iloc[:, 1]  # 第二列是基因名
data = df.iloc[:, 2:].values  # 第三列及之后是数据

# 通过 PCA 对数据进行降维
pca = PCA(n_components=5)
pca_result = pca.fit_transform(data)

# 聚类
kmeans = KMeans(n_clusters=3)
kmeans_result = kmeans.fit_predict(pca_result)

# 获取每个主成分的方差比例
var_ratio = pca.explained_variance_ratio_

# 绘制聚类图像
plt.figure(figsize=(10, 10))
for i in range(3):
    cluster_points = pca_result[kmeans_result == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c='C'+str(i), s=20, alpha=0.7, label=f'Cluster {i+1}')

# 绘制聚类边界
for i in range(3):
    cluster_points = pca_result[kmeans_result == i]
    centroid = kmeans.cluster_centers_[i]
    plt.plot(cluster_points[:, 0], cluster_points[:, 1], 'k.', markersize=1, alpha=0.2)
    plt.plot([centroid[0]], [centroid[1]], 'ko', markersize=10, alpha=1)

# 强调特定基因
emphasis_indices = np.where(df.iloc[:, 0] == 1)[0]
for index in emphasis_indices:
    x_offset = 0.21 if index % 2 == 0 else -0.21
    y_offset = 0.21 if index % 4 < 2 else -0.21
    plt.annotate(labels[index], (pca_result[index, 0], pca_result[index, 1]+y_offset), fontsize=6, color='red')

plt.xlabel("PC1: %.2f%%" % (var_ratio[0] * 100))

plt.xlabel("PC1: %.2f%%" % (var_ratio[0] * 100))
plt.ylabel("PC2: %.2f%%" % (var_ratio[1] * 100))
plt.title("PCA Cluster")
plt.legend()
plt.savefig('\\PCA_cluster.png', format='png', dpi=300)

# 对聚类结果进行分析，统计每个类别的样本数
class_count = {}
for i in range(3):
    class_count[i] = (kmeans_result == i).sum()
print(class_count)



# 将聚类结果写入 Excel 文件
writer = pd.ExcelWriter('\\cluster_results.xlsx')
for i in range(kmeans.n_clusters):
    # 获取每个类别中的基因名
    genes_in_cluster = labels[kmeans_result == i]
    # 将基因名转换为 DataFrame 对象
    genes_df = pd.DataFrame(genes_in_cluster, columns=['Genes in Cluster'])
    # 将 DataFrame 写入 Excel 文件中的不同 sheet
    genes_df.to_excel(writer, sheet_name=f'Cluster {i+1}')
writer.save()

