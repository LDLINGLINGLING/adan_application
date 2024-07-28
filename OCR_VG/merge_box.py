import json
from PIL import Image, ImageDraw
import time
import os
from sklearn.cluster import KMeans
import numpy as np
import cv2
import copy
from PIL import Image, ImageDraw, ImageFont
import math
import random


# 用于opencv查看标注
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "/Users/liudan/ai/simsun.ttc", textSize, encoding="utf-8"
    )
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# 计算两个矩形的交集占小矩形百分比
def calculate_intersection_percentage(rect1, rect2):
    def calculate_rectangle_area(rect):
        x1, y1 = rect[0]
        x2, y2 = rect[1]
        return abs((x2 - x1) * (y2 - y1))

    def calculate_intersection_area(rect1, rect2):
        x11, y11 = rect1[0]
        x12, y12 = rect1[1]
        x21, y21 = rect2[0]
        x22, y22 = rect2[1]

        x_left = max(x11, x21)
        y_top = max(y11, y21)
        x_right = min(x12, x22)
        y_bottom = min(y12, y22)

        if x_right < x_left or y_bottom < y_top:
            return 0

        return (x_right - x_left) * (y_bottom - y_top)

    # 计算两个矩形的面积
    area1 = calculate_rectangle_area(rect1)
    area2 = calculate_rectangle_area(rect2)

    # 计算两个矩形的交集面积
    intersection_area = calculate_intersection_area(rect1, rect2)

    # 确定较小矩形的面积
    smaller_area = min(area1, area2)

    # 计算交集面积占较小矩形面积的百分比
    percentage = (intersection_area / smaller_area) * 100 if smaller_area > 0 else 0

    return percentage


# 闭包函数使用聚类获取box的y轴阈值，用于决定是否切分
def get_threshold(boxes, text_list, width, height):
    y_gaps = []
    for i, box in enumerate(boxes):
        if i == 0:
            continue
        else:
            y_gap = abs(box[0][1] - boxes[i - 1][-1][1])
            y_gaps.append(y_gap)
    data = np.array(y_gaps)
    data = data.reshape(-1, 1)
    if len(y_gaps) == 1:
        y_threshold = boxes[0][-1][1] - boxes[0][0][1]
        x_threshold = int(width * 0.05)
        return y_threshold, x_threshold
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)
    centers = kmeans.cluster_centers_.flatten()

    # 获取数据点的簇标签
    labels = kmeans.labels_

    # 找出属于较小簇的数据点
    smaller_cluster_data = data[labels == np.argmin(centers)]

    # 计算较小簇中最大值
    y_threshold = int(np.max(smaller_cluster_data)) + 2
    x_threshold = int(width * 0.05)
    return y_threshold, x_threshold


# 获取多个矩形的最小包围矩形左上角和右下角坐标
def find_min_bounding_rectangle(polygons):
    # 初始化x和y的最小和最大值
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    # 遍历所有多边形
    for polygon in polygons:
        # 遍历一个多边形的所有顶点
        for point in polygon:
            x, y = point
            # 更新x和y的最小和最大值
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    # 返回最小包围矩形的左上角和右下角坐标
    return [[min_x, min_y], [max_x, max_y]]


# 进行融合的主函数
def merge_boxes(boxes, text_list, width, height):
    if len(boxes) == 1:
        return boxes, text_list
    
    # 获取y轴阈值和x轴阈值
    y_threshold, x_threshold = get_threshold(boxes, text_list, width, height)
    final_box = []
    temp_boxes = []
    temp_text = []
    final_text = []

    for index, box in enumerate(boxes):
        if index == 0:
            temp_boxes.append(box)
            temp_text.append(text_list[index])
        else:
            if (
                box[0][1] - boxes[index - 1][-1][1] < y_threshold
                and box[0][0] - boxes[index - 1][-1][0] < x_threshold
            ):
                if len(final_box) >= 1:
                    if (
                        calculate_intersection_percentage(
                            [box[0], box[-2]], final_box[-1]
                        )
                        < 80
                    ):
                        temp_boxes.append(box)
                        temp_text.append(text_list[index])
                    else:
                        temp_list = copy.deepcopy(final_box[-1])
                        final_box[-1] = find_min_bounding_rectangle([temp_list, box])
                        temp_string = final_text[-1] + text_list[index]
                        final_text[-1] = temp_string
                else:
                    temp_boxes.append(box)
                    temp_text.append(text_list[index])
            else:
                final_box.append(find_min_bounding_rectangle(temp_boxes))
                final_text.append("".join(temp_text))
                temp_boxes = []
                temp_text = []
                temp_boxes.append(box)
                temp_text.append(text_list[index])
            if index == len(boxes) - 1:
                assert len(temp_boxes) == len(temp_text)
                final_box.append(find_min_bounding_rectangle(temp_boxes))
                final_text.append("".join(temp_text))
    assert len(final_box) == len(final_text)
    return final_box, final_text


# 判断列表中四个坐标点是否构成矩形
def is_rectangle(points):
    # 提取所有x坐标和y坐标
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    # 检查x坐标和y坐标是否只有两种不同的值
    unique_x = len(set(x_coords)) == 2
    unique_y = len(set(y_coords)) == 2

    # 检查每对x坐标和y坐标是否恰好出现两次
    correct_x_count = all(x_coords.count(x) == 2 for x in set(x_coords))
    correct_y_count = all(y_coords.count(y) == 2 for y in set(y_coords))

    return unique_x and unique_y and correct_x_count and correct_y_count


# 获取格式化的json文件，按照官方格式，仅在输出中增加了<box>x1,y1,x2,y2</box>作为
def get_query_answer(boxes, text_list, height, width):
    assert len(boxes) == len(text_list)
    query_list = [
        "请帮我识别图片里的文字内容，并标出它们的确切位置。",
        "能否识别出图像上的文字，并告诉我它们所在的具体坐标？",
        "我需要你找出图片中所有的文字，并给出它们的位置信息。",
        "请分析这张图片，识别其中的文字，并指出每段文字的准确位置。",
        "能否帮忙识别下图中的文本，并提供它们的定位数据？",
        "请识别图像内的文字，并附上它们的坐标。",
        "我想知道图中文字的内容以及它们各自的位置。",
        "请读取图片上的文字，并记录下它们的精确位置。",
        "识别一下图片中的文字，并给出它们的坐标吧。",
        "请把图片里的文字找出来，并标注出它们的地点。",
    ]
    query = random.choice(query_list)
    format_answer = ""

    # 对坐标进行归一化
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = (
            int(box[0][0] * 1000 / width),
            int(box[0][1] * 1000 / height),
        
        int(box[1][0] * 1000 / width), int(box[1][1] * 1000 / height)
        )

        format_answer += "<ref>{text}<box>{x1}</box><box>{y1}</box><box>{x2}</box><box>{y2}</box></ref>".format(
            text=text_list[index], x1=x1, y1=y1, x2=x2, y2=y2
        )
    return query, format_answer

def load_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def draw(final_box,final_text,img):
    color = (0, 0, 255)
    # 绘制每个矩形
    for r_index,rectangle in enumerate(final_box):
        # PIL需要左上角和右下角的坐标来绘制矩形
        left_top = (int(rectangle[0][0]), int(rectangle[0][1]))
        right_bottom = (int(rectangle[1][0]), int(rectangle[1][1]))

        # 使用draw.rectangle()函数绘制矩形
        cv2.rectangle(img,left_top, right_bottom, color, thickness=2)
        text = final_text[r_index]  # 要显示的文字
        text = text[:5]+text[-5:] if len(text)>10 else text
        img = cv2ImgAddText(img, text, left_top[0], left_top[1], (0, 0 , 255), 30)

    # 在屏幕上显示带有矩形框的图片
    cv2.imshow('Image with Rectangles', img)
    cv2.waitKey(3500)  # 保持窗口打开5000毫秒，即5秒

    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()
    return
def save_to_json(output_data, output_path):
    random.shuffle(output_data)
    with open(
        os.path.join(output_path,"vg_box_train.json"), "w"
    ) as file:
        json.dump(output_data, file, ensure_ascii=False, indent=4)
    with open(
       os.path.join(output_path,"vg_box_test.json"), "w"
    ) as file:
        json.dump(output_data[-200:], file, ensure_ascii=False, indent=4)
    return 
def main():
    data=load_json(json_path)
    all_data_dict = data["data"]
    output_data = []
    index = 0
    for k, v in all_data_dict.items():
        out_dict = {}
        gt = v["gt"]
        boxes = [box["polygon"] for box in gt]
        if not all([is_rectangle(box) for box in boxes]):
            continue
        text_list = [box["text"] for box in gt]
        img = cv2.imread(k)
        height, width = img.shape[:2]
        final_box, final_text = merge_boxes(boxes, text_list, width, height)
        out_dict["id"] = str(index)
        out_dict["image"] = k
        query, format_answer = get_query_answer(final_box, final_text, height, width)
        out_dict["conversations"] = [
            {"content": "<image>\n{}".format(query), "role": "user"},
            {"content": format_answer, "role": "assistant"},
        ]
        output_data.append(out_dict)
        index += 1
        if check_box_text_pair:
            draw(final_box,final_text,img)     
    save_to_json(output_data, output_path)

if __name__ == "__main__":
    json_path="data_demo/img_gt.json" # 按照这个格式输入json文件，json也按照这个文件造数据
    output_path='data_demo/' # 最终数据文件输出路径
    check_box_text_pair=False # 是否可视化，可视化的话会打开一个窗口，显示你标注的框和文字
    main()