import numpy as np
from PIL import Image
import imutils
from imutils.perspective import four_point_transform
from imutils import contours
import os
import cv2
import numpy as np

def get_edge(img):
    """通过画面颜色确定车牌的左右边缘"""
    cv2.imshow("img", img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    counts = np.bincount(h.reshape(-1))
    h_channel[np.argmax(counts)] += 1
    # print(np.argmax(counts))
    Lower = np.array([75, 20, 20])  # 要识别颜色的下限
    Upper = np.array([130, 255, 255])  # 要识别的颜色的上限
    # mask是把HSV图片中在颜色范围内的区域变成白色，其他区域变成黑色
    mask = cv2.inRange(hsv, Lower, Upper)
    cv2.imshow("mask", mask)
    # cv2.waitKey(20000)
    # cv2.imwrite(os.path.join(result_dir, 'color_' + filelist[idx]), mask)
    mask_col_mean = np.mean(mask, 0)
    edge_ratio = 0.6
    for i in range(len(mask_col_mean)):
        if mask_col_mean[i] > edge_ratio * np.max(mask_col_mean):
            left_edge_x = i
            break
    for i in range(len(mask_col_mean)):
        if mask_col_mean[len(mask_col_mean) - 1 - i] > edge_ratio * np.max(mask_col_mean):
            right_edge_x = len(mask_col_mean) - i
            break
    cv2.line(mask, (left_edge_x, 5), (left_edge_x, 60), (0, 255, 0), 1)
    cv2.line(mask, (right_edge_x, 5), (right_edge_x, 60), (0, 255, 0), 1)
    # cv2.imwrite(os.path.join(result_dir, 'edge_' + filelist[idx]), mask)
    return left_edge_x, right_edge_x

def select_contours(img):
    """从图像中得到初步筛选的字符框"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #转灰度图
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,61,2)  #二值化
    kernel_morph = np.ones((3, 3), np.uint8)                                         #开运算

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_morph)
    cv2.imshow("opening", opening)
    # cv2.waitKey()
    opening, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    #寻找轮廓框

    # 根据框的宽高进行框的初步筛选
    selected_contours = []
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if h < 0.85 * h_img and h > 0.3 * h_img and h > w:
            if hierarchy[0][i][3] < 0:
                selected_contours.append([x, y, w, h])

    return selected_contours

def reg_line(selected_contours):
    """通过中间可靠性较高的框的中心回归出一条直线"""
    # 将框从左到右排序，并在回归直线时舍去最左最右的两个框
    sorted_contours = sorted(selected_contours, key=lambda k: k[0])
    reg_contours = sorted_contours.copy()
    if len(reg_contours) > 2:
        reg_contours.pop()
        reg_contours.pop(0)
    reg_contours_arr = np.array(reg_contours)

    # 使用各个框左上、右上、左下的坐标，求取三次斜率，并取平均
    # 左上
    reg_x = np.transpose(reg_contours_arr[:, 0])
    reg_y = np.transpose(reg_contours_arr[:, 1])
    A_xy = np.vstack([reg_x, np.ones(len(reg_x))]).T
    m_xy, c_xy = np.linalg.lstsq(A_xy, reg_y, rcond=-1)[0]

    # 右上
    reg_xw = np.transpose(reg_contours_arr[:, 0] + reg_contours_arr[:, 2])
    A_xwy = np.vstack([reg_xw, np.ones(len(reg_xw))]).T
    m_xwy, c_xwy = np.linalg.lstsq(A_xwy, reg_y, rcond=-1)[0]

    # 左下
    reg_yh = np.transpose(reg_contours_arr[:, 1] + reg_contours_arr[:, 3])
    m_xyh, c_xyh = np.linalg.lstsq(A_xy, reg_yh, rcond=-1)[0]

    # 取平均
    m_mean = np.mean([m_xy, m_xwy, m_xyh])

    # 确定回归直线的位置，让回归直线经过各框中心点的均值
    # 中心
    reg_xc = np.transpose(reg_contours_arr[:, 0] + 0.5 * reg_contours_arr[:, 2])
    reg_yc = np.transpose(reg_contours_arr[:, 1] + 0.5 * reg_contours_arr[:, 3])
    A_c = np.vstack([reg_xc, np.ones(len(reg_xc))]).T
    m_c, c_c = np.linalg.lstsq(A_c, reg_yc, rcond=-1)[0]

    return m_mean, c_c

def exclude_contours(selected_contours, m_mean, c_c):
    """将框按照其中心到回归直线的距离，选出置信度最高的9个框"""
    selected_contours_arr = np.array(selected_contours)

    # 列4：框中心点的x坐标
    selected_contours_arr_xc = selected_contours_arr[:, 0] + 0.5 * selected_contours_arr[:, 2]
    selected_contours_arr = np.column_stack((selected_contours_arr, selected_contours_arr_xc))

    # 列5：框中心点的y坐标
    selected_contours_arr_yc = selected_contours_arr[:, 1] + 0.5 * selected_contours_arr[:, 3]
    selected_contours_arr = np.column_stack((selected_contours_arr, selected_contours_arr_yc))

    # 列6：框中心点到回归直线的距离
    selected_contours_arr_distance = abs(m_mean * selected_contours_arr[:, 4] + c_c - selected_contours_arr[:, 5]) / np.sqrt(m_mean * m_mean + 1)
    selected_contours_arr = np.column_stack((selected_contours_arr, selected_contours_arr_distance))
    sorted_contours = sorted(selected_contours_arr, key=lambda k: k[6])

    # 列7：将框按照其中心到回归直线的距离，从小到大排序
    sorted_contours_arr = np.array(sorted_contours)
    top9_contours = sorted_contours_arr[0:9, :]
    # top9_contours = sorted_contours_arr

    return top9_contours

if __name__ == '__main__':
    # import sys
    # ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
    # if ros_path in sys.path:
    #     sys.path.remove(ros_path)
    #     import cv2
    #     sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

    # 将这个文件和train-data文件夹放在同一文件夹下
    data_dir = './train-data/200fail'
    result_dir = './train-data/result'
    filelist = os.listdir(data_dir)

    count_completed = 0     # 判断是否得到了9个以上的框
    count_failed = 0
    h_channel = np.zeros(256)
    for idx in range(len(filelist)):
    # for idx in range(1):
    # idx = 3
    # if idx > 0:
        file_name = os.path.join(data_dir, filelist[idx])
        # file_name = os.path.join(data_dir, '1.jpg')

        # 用CV2读入
        img = cv2.imread(file_name)
        h_img = img.shape[0]
        w_img = img.shape[1]

        selected_contours = select_contours(img)  # 初步筛选
        # if 1:
        if len(selected_contours) >= 9:             # 判断是否得到了9个以上的框
            print(idx, "more than 9")
            count_completed += 1
            m_mean, c_c = reg_line(selected_contours)   # 使用可靠性比较高的框的中心点，回归出一条直线
            top9_contours = exclude_contours(selected_contours, m_mean, c_c)    # 挑选中心最靠近直线的9个框

            # 将9个框依次画在原始图片中
            for i in range(len(top9_contours)):
                x, y = int(top9_contours[i][0]), int(top9_contours[i][1])
                w, h = int(top9_contours[i][2]), int(top9_contours[i][3])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # 输出结果图片
            # cv2.imwrite(os.path.join(result_dir, filelist[idx]), img)
            cv2.imshow("window", img)
            # cv2.destroyAllWindows()
        cv2.waitKey(1000)
    print("Completed:", count_completed, "Failed:", count_failed)