import pandas as pd
import os

# 定义目录和输出文件名
folder_path = '你的文件夹路径'  # 例如: '/path/to/csv/folder'
output_file = 'combined.csv'

# 定义你想要合并的文件名
desired_files = ['file1.csv', 'file2.csv', 'file3.csv', 'file4.csv']  # 根据实际情况修改

# 在目录中查找和 desired_files 匹配的文件
all_files = [f for f in os.listdir(folder_path) if f in desired_files]

# 如果没有找到匹配的.csv文件，打印消息并退出
if not all_files:
    print("没有找到匹配的.csv文件")
    exit()

# 创建一个空的 DataFrame 用于存储数据
all_data = pd.DataFrame()

# 循环遍历每个文件，读取内容并追加到 all_data DataFrame
for file in all_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)  
    
    # 确保
