import tkinter as tk
from tkinter import filedialog
import os
import sys

class GUI:
    def __init__(self, root):
        self.root = root
        self.text = tk.Text(root)
        self.text.pack()
        sys.stdout = self

    def write(self, message):
        self.text.insert(tk.END, message)

    def flush(self):
        pass  

def upload_file1():
    file_path = filedialog.askopenfilename()
    file_label1.config(text=file_path)

def upload_file2():
    file_path = filedialog.askopenfilename()
    file_label2.config(text=file_path)

def upload_file3():
    file_path = filedialog.askopenfilename()
    file_label3.config(text=file_path)

def choose_output_dir():
    global output_dir
    output_dir = filedialog.askdirectory()
    print(output_dir)  # 输出选择的文件夹位置



#1）BLAST建库
from Bio.Blast.Applications import NcbimakeblastdbCommandline
def create_blast_db(file_path, output_dir):
    db_name = file_path.split("/")[-1].split(".")[0] + "_blastdb"
    cmd = NcbimakeblastdbCommandline(
        input_file=file_path,
        dbtype="prot",
        out=output_dir + "/" + db_name,
    )
    stdout, stderr = cmd()
    print(stdout)
    print(stderr)



#2）进行正向BLAST并输出结果
from Bio.Blast.Applications import NcbiblastpCommandline
def run_blastp(query_file, subject_file, output_dir):
    out_file = f"{output_dir}/a_to_b.blast"
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
def run_blastp_1(query_file, subject_file, output_dir):
    out_file = f"{output_dir}/b_to_a.blast"
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


#作为分析的核心程序，涵盖所有分析的函数
def analyze_files():
    file1_path = file_label1.cget("text")
    file2_path = file_label2.cget("text")
    file3_path = file_label3.cget("text")

    create_blast_db(file1_path,output_dir)
    print("The library of species A was successfully established")

    create_blast_db(file2_path,output_dir)
    print("The library of species B was successfully established")

    run_blastp(file3_path,file2_path,output_dir)
    print("Forward blast has been completed")

    filter_blast_results(f"{output_dir}/a_to_b.blast",f"{output_dir}/a_to_b_filtered.blast")
    print("BLAST result filtering complete, best hits have been selected")

    filter_fasta(f"{output_dir}/a_to_b_filtered.blast",file2_path,f"{output_dir}/b.fa")
    print("Filtered fasta data of B was selected")

    run_blastp_1(f"{output_dir}/b.fa",file1_path,output_dir)
    print("Reverse blast has been completed")

    filter_blast_results(f"{output_dir}/b_to_a.blast",f"{output_dir}/b_to_a_filtered.blast")
    print("BLAST result filtering complete, best hits have been selected")

    write_mutual_best_hit(f"{output_dir}/a_to_b_filtered.blast",f"{output_dir}/b_to_a_filtered.blast",f"{output_dir}/mutual_best_hit.csv")
    print("Mutual best hits complete")

#可视化界面的实现
root = tk.Tk()
root.title("Mutual Best-hits of Protein")
gui = GUI(root)

file_label1 = tk.Label(root, text="Species A Protein FASTA")
file_label1.pack()

upload_button1 = tk.Button(root, text="Upload file1", command=upload_file1)
upload_button1.pack()

file_label2 = tk.Label(root, text="Species B Protein FASTA")
file_label2.pack()

upload_button2 = tk.Button(root, text="Upload file2", command=upload_file2)
upload_button2.pack()

file_label3 = tk.Label(root, text="protein to be analyzed")
file_label3.pack()

upload_button3 = tk.Button(root, text="Upload file3", command=upload_file3)
upload_button3.pack()

# 添加一个按钮，点击后选择输出文件夹位置
button = tk.Button(root, text="Select output folder", command=choose_output_dir)
button.pack()

confirm_button = tk.Button(root, text="Comfirm", command=analyze_files)
confirm_button.pack()

root.mainloop()