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