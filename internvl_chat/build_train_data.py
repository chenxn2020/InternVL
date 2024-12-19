import json
import os
data_path = '/cpfs01/user/caixinyu/markdownGenerate/test_bbox'

save_prefix = '/cpfs01/user/chenxiangnan/InternVL_SEG/internvl_chat/shell/data'

def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    result = {}
    for filename in os.listdir(data_path):
        if filename.endswith(".jsonl"):  # 确保只处理 JSONL 文件
            file_path = os.path.join(data_path, filename)            
            # 读取 JSONL 文件内容
            with open(file_path, "r", encoding="utf-8") as f:
                data = f.readlines()
        keys = os.path.splitext(filename)[0]
        result[keys] = dict(
            root = '/',
            annotation = file_path,
            data_augment = False,
            repeat_time = 1,
            length = len(data),
        )
    save_name = 'docg_seg.json'
    output_path = os.path.join(save_prefix, save_name)
    save_json(result, output_path)