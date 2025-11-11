import json
import os
from tqdm import tqdm


def process_file(file_path):
    # 读取文件
    with open(file_path, 'r') as file:
        data = json.load(file)

    # 初始化进度条
    progress_bar = tqdm(data, desc=f"Processing {os.path.basename(file_path)}")

    # 修改 "length" 的值
    for item in progress_bar:
        if "length" in item:
            if item["length"] > 1:
                item["length"] -= 1
            else:
                # 如果 length 的值等于 0 或 1，跳过并打印出来
                progress_bar.write(f"Skipped item with length {item['length']}")

    # 保存修改后的文件
    file_name, file_extension = os.path.splitext(file_path)
    train_file_path = f"{file_name}_train{file_extension}"
    val_file_path = f"{file_name}_val{file_extension}"

    with open(train_file_path, 'w') as train_file:
        json.dump(data, train_file, indent=4)

    with open(val_file_path, 'w') as val_file:
        json.dump(data, val_file, indent=4)


# 指定目录
directory = "/home/gaoxiang12/aaai/live-rl/TADT-CSA/dt_format_datasets"

# 遍历目录下的文件
for file_name in os.listdir(directory):
    file_path = os.path.join(directory, file_name)
    if file_name.endswith(".json"):
        process_file(file_path)
        print(f"Processed {file_name}")