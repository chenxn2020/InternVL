import json
import os
data_path = '/cpfs01/user/caixinyu/markdownGenerate/test_bbox'

save_prefix = './shell/data'

def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)
def save_jsonl(data, output_path):
    with open(output_path, "w", encoding="utf-8") as json_file:
        for item in data:
            json_line = json.dumps(data, ensure_ascii=False)
            json_file.write(json_line + "\n")
def build_train_data():
    result = {}
    for filename in os.listdir(data_path):
        if filename.endswith(".jsonl"):  # 确保只处理 JSONL 文件
            file_path = os.path.join(data_path, filename)            
            # 读取 JSONL 文件内容
            with open(file_path, "r", encoding="utf-8") as f:
                data = f.readlines()
        keys = os.path.splitext(filename)[0]
        if keys != 'color_block_content_qa':
            continue
        
        result[keys] = dict(
            root = '/',
            annotation = file_path,
            data_augment = False,
            repeat_time = 1,
            length = 20000,
        )
    # save_name = 'docg_seg_color.json'
    save_name = 'color_train.json'
    output_path = os.path.join(save_prefix, save_name)
    save_json(result, output_path)

if __name__ == '__main__':
    #build color test
    path = '/cpfs01/user/caixinyu/markdownGenerate/test_bbox/color_block_content_qa.jsonl'
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    test_data = data[-500:]
    from IPython import embed; embed(); exit()
    save_jsonl(test_data, os.path.join(save_prefix, 'color_test.jsonl'))