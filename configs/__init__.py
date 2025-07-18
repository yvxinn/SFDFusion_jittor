"""
配置处理工具
====================================

本脚本提供了一个工具类 `ConfigDict` 和一个辅助函数 `from_dict`，
用于将从YAML文件中加载的标准字典（dict）递归地转换为一个可以
通过属性（attribute）方式访问其键值的配置对象。

例如，`config['dataset']['name']` 可以被更方便地写作
`config.dataset.name`。
"""

class ConfigDict(dict):
    """
    一个自定义的字典类，允许通过属性方式访问其键值。
    例如, `d.key` 等价于 `d['key']`。
    """
    # 将属性设置操作 `obj.key = value` 映射到字典的 `__setitem__` 方法 `obj['key'] = value`
    __setattr__ = dict.__setitem__
    # 将属性获取操作 `obj.key` 映射到字典的 `__getitem__` 方法 `obj['key']`
    __getattr__ = dict.__getitem__


def from_dict(obj: dict) -> ConfigDict:
    """
    递归地将一个标准字典及其所有嵌套的字典，转换为 `ConfigDict` 对象。

    Args:
        obj (dict): 输入的原始字典。

    Returns:
        ConfigDict: 一个可以通过属性访问的配置对象。
    """
    # 如果输入不是字典，直接返回原对象
    if not isinstance(obj, dict):
        return obj
    # 递归转换
    d = ConfigDict()
    for k, v in obj.items():
        d[k] = from_dict(v)
    return d
