import json
import torch
def print_model_hierarchy(module, prefix=''):
    """
    递归打印模型的子模块及其属性名。

    Args:
        module (torch.nn.Module): 要遍历的模型模块。
        prefix (str): 用于显示层级关系的前缀。
    """
    for name, submodule in module.named_children():
        print(f"{prefix}{name}: {type(submodule).__name__}")
        print_model_hierarchy(submodule, prefix=prefix + '  ')


def compare_dictionaries(dict1, dict2, atol=1e-6):
    if dict1.keys() != dict2.keys():
        print("The dictionaries have different keys.")
        return False

    for key in dict1:
        value1 = dict1[key]
        value2 = dict2[key]

        if type(value1) != type(value2):
            print(f"Type mismatch for key '{key}': {type(value1)} vs {type(value2)}")
            return False

        if isinstance(value1, torch.Tensor):
            if not torch.allclose(value1, value2, atol=atol):
                print(f"Tensor mismatch for key '{key}'")
                return False

        elif isinstance(value1, dict):
            if not compare_dictionaries(value1, value2, atol):
                return False

        elif value1 != value2:
            print(f"Value mismatch for key '{key}': {value1} vs {value2}")
            return False

    return True

if __name__ == "__main__":
    from IPython import embed; embed(); exit()
    # 加载并比较
    dict1 = torch.load("seg.pth")
    dict2 = torch.load("mag.pth")

    if compare_dictionaries(dict1, dict2):
        print("The dictionaries are identical!")
    else:
        print("The dictionaries are different.")