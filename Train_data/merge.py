import json

def merge_sft_files(file1_path, file2_path, output_file_path):
    """
    合并两个 SFT 格式的 JSON 文件并保存到指定路径。

    参数：
        file1_path (str): 第一个 JSON 文件路径。
        file2_path (str): 第二个 JSON 文件路径。
        output_file_path (str): 合并后保存的文件路径。
    """
    # 读取 JSON 文件
    def read_json(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 写入 JSON 文件
    def write_json(data, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # 合并两个文件的内容
    data1 = read_json(file1_path)
    data2 = read_json(file2_path)
    merged_data = data1 + data2

    # 写入到输出文件
    write_json(merged_data, output_file_path)

    print(f"合并完成，结果已保存到 {output_file_path}")
    
def merge_jsonl_files(file1_path, file2_path, output_file_path):
    """
    合并两个 JSONL 文件并保存到指定路径。

    参数：
        file1_path (str): 第一个 JSONL 文件路径。
        file2_path (str): 第二个 JSONL 文件路径。
        output_file_path (str): 合并后保存的文件路径。
    """
    # 读取 JSONL 文件
    def read_jsonl(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    # 写入 JSONL 文件
    def write_jsonl(data, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 合并两个文件的内容
    data1 = read_jsonl(file1_path)
    data2 = read_jsonl(file2_path)
    merged_data = data1 + data2

    # 写入到输出文件
    write_jsonl(merged_data, output_file_path)

    print(f"合并完成，结果已保存到 {output_file_path}")

if __name__ == "__main__":
    data_type = ["train_rl","test_rl"]
    for data in data_type:
        # 示例调用
        file1 = f"/nfsdata4/wengtengjin/oddgrid_task/OddGridBench_clean/IOL_type/train/{data}_data.jsonl"  # 替换为实际文件1路径
        file2 = f"/nfsdata4/wengtengjin/oddgrid_task/OddGridBench_clean/SOI_type/train/{data}_data.jsonl"  # 替换为实际文件2路径
        output_file = f"{data}_data.jsonl"  # 替换为保存路径
        
        

        merge_jsonl_files(file1, file2, output_file)
        
    data_type = ["train_sft","test_sft"]
    for data in data_type:
        # 示例调用
        file1 = f"/nfsdata4/wengtengjin/oddgrid_task/OddGridBench_clean/IOL_type/train/{data}_qa.json"  # 替换为实际文件1路径
        file2 = f"/nfsdata4/wengtengjin/oddgrid_task/OddGridBench_clean/SOI_type/train/{data}_qa.json"  # 替换为实际文件2路径
        output_file = f"{data}_qa.json"  # 替换为保存路径
        
        

        merge_sft_files(file1, file2, output_file)