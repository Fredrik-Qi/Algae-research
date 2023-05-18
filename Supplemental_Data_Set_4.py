import os
import pymol
from pymol import cmd

# 启动PyMOL会话
pymol.finish_launching()

# 指定PDB文件所在的文件夹路径
pdb_folder = r""

# 获取文件夹中的所有PDB文件
pdb_files = [f for f in os.listdir(pdb_folder) if f.endswith(".pdb")]

# 遍历每个PDB文件
for pdb_file in pdb_files:
    # 构建PDB文件的完整路径
    pdb_path = os.path.join(pdb_folder, pdb_file)

    # 清空之前的场景
    cmd.reinitialize()

    # 加载PDB文件
    cmd.load(pdb_path)

    # 设置视角和显示选项
    cmd.viewport(800, 600)
    cmd.hide('everything')
    cmd.show('cartoon')

    # 设置背景颜色为白色
    cmd.bg_color('white')

    # 设置渲染参数
    cmd.set('spec_reflect', 0)
    cmd.set('ray_trace_mode', 0)
    cmd.set('ray_shadow', 0)
    cmd.set('fog', 0)

    # 设置颜色映射
    cmd.set_color('high_lddt_c', [0, 0.325490196078431, 0.843137254901961])
    cmd.set_color('normal_lddt_c', [0.341176470588235, 0.792156862745098, 0.976470588235294])
    cmd.set_color('medium_lddt_c', [1, 0.858823529411765, 0.070588235294118])
    cmd.set_color('low_lddt_c', [1, 0.494117647058824, 0.270588235294118])

    # 根据置信度进行着色
    cmd.color('high_lddt_c', '(b > 90)')
    cmd.color('normal_lddt_c', '(b < 90 and b > 70)')
    cmd.color('medium_lddt_c', '(b < 70 and b > 50)')
    cmd.color('low_lddt_c', '(b < 50)')

    # 构建输出PNG文件名
    output_path = os.path.join(pdb_folder, f"{os.path.splitext(pdb_file)[0]}_output.png")

    # 渲染图像并保存为PNG文件
    cmd.png(output_path, width=800, height=600, dpi=300, ray=1)

# 关闭PyMOL会话
pymol.cmd.quit()
