# -*- coding: utf-8 -*- 
# @Time : 2021/6/5 10:50
# @Author : hangzhouwh 
# @Email: hangzhouwh@gmail.com
# @File : generate_hoc_images.py 
# @Software: PyCharm


import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
import colorsys
import random


def load_json(filepath):
    file = open(filepath, 'rb')
    data = json.load(file)
    return data


def write_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f)


# 每种代码块对应的编号
checkFeature = {}


# 处理HOC18
def generate_ast(DG, parent, input_json, depth):
    if isinstance(input_json, dict):  # 结点
        id = int(input_json['id'])
        optype = input_json['type']

        if optype not in checkFeature:  # 为HOC18中每种代码块赋值一个编号
            if checkFeature:
                num = max(checkFeature.values()) + 1
            else:
                num = 0
            checkFeature[optype] = num

        feature = checkFeature[optype]

        DG.add_node(id, feature=feature, opcode=optype, depth=depth)
        if input_json.get('children') is None:  # 没有孩子了
            if parent is not None:
                DG.add_edge(parent, id)
        else:  # 有孩子
            child_lst = input_json['children']
            if isinstance(child_lst, dict):
                generate_ast(DG, id, child_lst, depth + 1)
            elif isinstance(child_lst, list):
                for json_array in child_lst:
                    generate_ast(DG, id, json_array, depth + 1)
                else:  # 没有孩子
                    if parent is not None:
                        DG.add_edge(parent, id)
    elif isinstance(input_json, list):  # 结点lst
        for input_json_array in input_json:
            generate_ast(DG, parent, input_json_array, depth + 1)


# 打印图——原始图和带上属性值的图
def plotGraphFeature(graph):
    pos = nx.shell_layout(graph)
    nx.draw(graph, pos)
    node_labels = nx.get_node_attributes(graph, 'feature', )
    nx.draw_networkx_labels(graph, pos, labels=node_labels)
    plt.show()

    nx.draw(graph, pos)
    node_labels = nx.get_node_attributes(graph, 'opcode')
    nx.draw_networkx_labels(graph, pos, labels=node_labels)
    plt.show()


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors


def color(value):
    digit = list(map(str, range(10))) + list("ABCDEF")
    if isinstance(value, tuple):
        string = '#'
        for i in value:
            a1 = i // 16
            a2 = i % 16
            string += digit[a1] + digit[a2]
        return string
    elif isinstance(value, str):
        a1 = digit.index(value[1]) * 16 + digit.index(value[2])
        a2 = digit.index(value[3]) * 16 + digit.index(value[4])
        a3 = digit.index(value[5]) * 16 + digit.index(value[6])
        return a1, a2, a3


colors = list(map(lambda x: color(tuple(x)), ncolors(16)))

fileList = os.listdir("D:/PYc/Pytorch/hoc_scode/")
fileId = []
for filename in fileList:
    filePre = filename.split('.')[0]
    if filePre.isdigit():
        fileId.append(int(filePre))
fileId.sort()


# HOC18
graphs = []

for i in tqdm(fileId):
    path = "D:/PYc/Pytorch/hoc_scode/" + str(i) + ".json"
    data_json = load_json(path)
    DG = nx.Graph(name="AST"+str(i))
    generate_ast(DG, None, data_json, 0)
    graphs.append(DG)

labelpath = 'D:/PYc/Pytorch/graph_labels.txt'
labels = []

with open(labelpath, 'r') as file:
    for line in file:
        labels.append(int(line.split()[0]))

index = 0

for graph in tqdm(graphs):
    DG = graph
    root = list(DG.nodes())[0]
    dfs_seqs = list(nx.dfs_tree(DG, root))

    array = np.ndarray((250, 250, 3), np.uint8)
    array[:, :, 0] = 255
    array[:, :, 1] = 255
    array[:, :, 2] = 255
    image = Image.fromarray(array)
    draw = ImageDraw.Draw(image)

    num = 0
    for node in dfs_seqs:
        draw.rectangle((DG.nodes()[node]['depth']*10, num*10, DG.nodes()[node]['depth']*10 + 70, num*10 + 10),
                       colors[checkFeature[DG.nodes()[node]['opcode']]], 'black')
        num += 1

    image.save('D:/PYc/Test-images/' + str(index) + '_' + str(labels[index]) + '.jpg')
    index += 1