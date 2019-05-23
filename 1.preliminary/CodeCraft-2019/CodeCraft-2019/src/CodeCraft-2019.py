import logging
import sys
from collections import defaultdict
from heapq import *
import re
import numpy as np


logging.basicConfig(level=logging.DEBUG,
                    filename='../logs/CodeCraft-2019.log',
                    format='[%(asctime)s] %(levelname)s [%(funcName)s: %(filename)s, %(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filemode='a')

# carRoute: [car_id, startTime, start, ..., end, 0, 0, ...]
# graph: [start, end, length, road_id, graph_id]
# carData: [car_id, start, end, speed, planTime]


# 超参数

w_channel_1 = 0  # 车道数对道路cost的影响（比赛中设置下列参数，该项影响不起作用）
w_channel_2 = 1  # 车道数对道路cost的影响（比赛中设置下列参数，该项影响不起作用）
w_roadheat = 0.02  # 道路拥挤程度的权重

run_time = 100  # 预计一辆车在路上行驶的时间（按照系统总调度时间与最后一辆车的发车时间之差预估）
num_heatmap = 5000  # 给heatmap预先分配的空间（足够大就行）
num_car = 35  # 每个时间片发的车辆数（数值越高，总调度时间越短）

reseek_ratio = 0.01  # 重新规划的比例
w_roadheat_reseek = 0.05  # 重新规划中，道路拥挤程度的权重


def weight_direction(direction):
    '''
    车辆在路口行驶方向的权重，仅在dijkstra算法中使用
    '''
    return 1.1 * direction


def dataProcess(carPath, crossPath, roadPath):
    carData = []
    crossData = []
    roadData = []
    with open(carPath, 'r') as lines:
        for line in lines:
            line = line.split(',')
            if re.findall("\d+", line[0]) != []:
                line[0] = re.findall("\d+", line[0])[0]
            if re.findall("\d+", line[-1]) != []:
                line[-1] = re.findall("\d+", line[-1])[0]
            # for i in range(len(line)):
            #     line[i] = int(line[i].strip())
            carData.append(line)
    with open(roadPath, 'r') as lines:
        for line in lines:
            line = line.split(',')
            if re.findall("\d+", line[0]) != []:
                line[0] = re.findall("\d+", line[0])[0]
            if re.findall("\d+", line[-1]) != []:
                line[-1] = re.findall("\d+", line[-1])[0]
            roadData.append(line)
    with open(crossPath, 'r') as lines:
        for line in lines:
            line = line.split(',')
            if re.findall("\d+", line[0]) != []:
                line[0] = re.findall("\d+", line[0])[0]
            if re.findall("\d+", line[-1]) != []:
                line[-1] = re.findall("\d+", line[-1])[0]
            crossData.append(line)

    carData = carData[1:]
    for i in range(len(carData)):
        for j in range(len(carData[i])):
            carData[i][j] = int(carData[i][j].strip())
    roadData = roadData[1: ]
    for i in range(len(roadData)):
        for j in range(len(roadData[i])):
            roadData[i][j] = int(roadData[i][j].strip())
    crossData = crossData[1: ]
    for i in range(len(crossData)):
        for j in range(len(crossData[i])):
            crossData[i][j] = int(crossData[i][j].strip())
    return carData, crossData, roadData


def dijkstra(edges, cross_link, f, t):
    """
    dijkstra算法规划路径

    return : 整条路径的cost + 路径经过的路口id
    """

    g = defaultdict(list)

    for i in range(edges.shape[0]):
        g[str(int(edges[i, 0]))].append((edges[i, 2], str(int(edges[i, 1]))))  # g[start] = (cost, end)

    q, seen, mins = [(0, f, ())], set(), {f: 0}
    while q:
        (cost, v1, path) = heappop(q)
        if v1 not in seen:
            seen.add(v1)
            path = (v1, path)
            if v1 == t:
                return (cost, path)

            for c, v2 in g.get(v1, ()):
                if v2 in seen: continue
                prev = mins.get(v2, None)
                if path[1] != ():
                    s = int(path[1][0])
                    direction = cross_link[cross_link[:, 0] == s]
                    direction = direction[direction[:, 1] == int(v1)]
                    direction = direction[direction[:, 2] == int(v2)]
                    direction = direction[0][3]
                    w_direct = weight_direction(direction)
                else:
                    w_direct = 1
                next = cost + w_direct * c
                if prev is None or next < prev:
                    mins[v2] = next
                    heappush(q, (next, v2, path))

    return float("inf")


def reseek(carRoute, carNum, car, graph, heat_map, cross_link):

    reseek_num = round(reseek_ratio * carNum)
    reseek_index = np.random.permutation(np.arange(carNum))[0: reseek_num]

    length = graph[:, 2]

    time_slice = carNum // num_car + 10
    cost_heat = heat_map[time_slice, :]

    for i in range(reseek_num):

        car_limit = np.ones((graph.shape[0], 2))
        car_limit[:, 0] *= car[reseek_index[i], 3]
        car_limit[:, 1] = graph[:, 5]
        speed_actual = car_limit.min(1)
        cost_length = np.ceil(length / speed_actual)

        cost = cost_length + w_roadheat_reseek * cost_heat
        cost = np.clip(cost, 0.5, 10000000)

        edges = np.zeros(graph.shape)
        edges[:, 0:2] = graph[:, 0:2]
        edges[:, 2] = cost

        result = dijkstra(edges, cross_link, str(car[reseek_index[i], 1]), str(car[reseek_index[i], 2]))

        sumarize = []
        while result[1] != ():
            sumarize.append(int(result[0]))
            if result[1] != ():
                result = result[1]
        sumarize.append(int(result[0]))  # cost, end, ..., start

        lengthSumarize = len(sumarize)
        carRouteTemp = np.zeros(carRoute.shape[1]).astype(int)
        carRouteTemp[0] = carRoute[reseek_index[i], 0]
        carRouteTemp[1] = carRoute[reseek_index[i], 1]

        # deduce road_heat
        carRoute_old = carRoute[reseek_index[i], :]
        cost_each_route = np.zeros(np.argwhere(carRoute_old != 0).max() + 1 - 2)
        route_each_route = np.zeros(np.argwhere(carRoute_old != 0).max() + 1 - 2)
        s = car[reseek_index[i], 1]  # start
        for j in range(2, np.argwhere(carRoute_old != 0).max() + 1):
            roadchoosen = graph[graph[:, 3] == carRoute_old[j]]
            roadchoosen = roadchoosen[roadchoosen[:, 0] == s]
            s = roadchoosen[0, 1]
            cost_each_route[j - 2] = cost_length[roadchoosen[0, 4]]
            route_each_route[j - 2] = roadchoosen[0, 4]
        time_slice = carRoute_old[1]
        time_start = time_slice
        for j in range(np.argwhere(carRoute_old != 0).max() + 1 - 2):
            time_end = int(time_start + cost_each_route[j])
            change_start = max((3 * time_start - time_end) // 2, 0)
            change_end = min((3 * time_end - time_start) // 2, len(heat_map))
            # heat_map[change_start:change_end, :] += road_heat[change_start:change_end, :]
            heat_map[change_start:change_end, int(route_each_route[j])] -= 1
            time_start = time_end

        # renew carRoute and road_heat
        cost_each_route = np.zeros(lengthSumarize - 2)
        route_each_route = np.zeros(lengthSumarize - 2)
        for j in range(1, lengthSumarize - 1):
            roadchoosen = graph[graph[:, 0] == sumarize[lengthSumarize - j]]  # find the start
            roadchoosen = roadchoosen[roadchoosen[:, 1] == sumarize[lengthSumarize - j - 1]]  # find the end
            carRouteTemp[j + 1] = roadchoosen[0, 3]
            cost_each_route[j - 1] = cost_length[roadchoosen[0, 4]]
            route_each_route[j - 1] = roadchoosen[0, 4]
        carRoute[reseek_index[i], :] = carRouteTemp

        time_slice = carRouteTemp[1]
        time_start = time_slice
        for j in range(lengthSumarize - 2):
            time_end = int(time_start + cost_each_route[j])
            change_start = max((3 * time_start - time_end) // 2, 0)
            change_end = min((3 * time_end - time_start) // 2, len(heat_map))
            # heat_map[change_start:change_end, :] += road_heat[change_start:change_end, :]
            heat_map[change_start:change_end, int(route_each_route[j])] += 1
            time_start = time_end

    return carRoute, heat_map


def caculate_cross_link(cross, graph):
    '''
    计算相邻三个路口的关系

    由于在官方调度器中有 直行的优先权 > 左转优先权 > 右转优先权 的规则，所以希望在规划路径中路径尽可能的走直线。从而需要在dijkstra算法中搜索路径的过程中加入有关转向的cost。
    cross_link记录三个相邻路口的关系，cross_link每一行代表一个转向的记录： 例如给出路线 路口A->路口B->路口C， 可以判断在路口B的行驶方向（直行记为0，左转记为1，右转记为2）。所以cross_link的每一行均为 [A, B, C, 行驶方向]
    '''
    cross_link = np.zeros((12 * cross.shape[0], 4)).astype(int)
    for i in range(cross.shape[0]):
        v1 = cross[i, 0]
        if cross[i, 1] != -1:
            s1 = graph[graph[:, 3] == cross[i, 1], :]
            if s1[0][1] != v1:
                s1 = s1[0][1]
            else:
                s1 = s1[0][0]

        if cross[i, 2] != -1:
            s2 = graph[graph[:, 3] == cross[i, 2], :]
            if s2[0][1] != v1:
                s2 = s2[0][1]
            else:
                s2 = s2[0][0]

        if cross[i, 3] != -1:
            s3 = graph[graph[:, 3] == cross[i, 3], :]
            if s3[0][1] != v1:
                s3 = s3[0][1]
            else:
                s3 = s3[0][0]

        if cross[i, 4] != -1:
            s4 = graph[graph[:, 3] == cross[i, 4], :]
            if s4[0][1] != v1:
                s4 = s4[0][1]
            else:
                s4 = s4[0][0]

        if cross[i, 1] != -1:
            if cross[i, 2] != -1:
                cross_link[i * 12 + 0, :] = [s1, v1, s2, 1]
            if cross[i, 3] != -1:
                cross_link[i * 12 + 1, :] = [s1, v1, s3, 0]
            if cross[i, 4] != -1:
                cross_link[i * 12 + 2, :] = [s1, v1, s4, 2]

        if cross[i, 2] != -1:
            if cross[i, 3] != -1:
                cross_link[i * 12 + 3, :] = [s2, v1, s3, 1]
            if cross[i, 4] != -1:
                cross_link[i * 12 + 4, :] = [s2, v1, s4, 0]
            if cross[i, 1] != -1:
                cross_link[i * 12 + 5, :] = [s2, v1, s1, 2]

        if cross[i, 3] != -1:
            if cross[i, 4] != -1:
                cross_link[i * 12 + 6, :] = [s3, v1, s4, 1]
            if cross[i, 1] != -1:
                cross_link[i * 12 + 7, :] = [s3, v1, s1, 0]
            if cross[i, 2] != -1:
                cross_link[i * 12 + 8, :] = [s3, v1, s2, 2]

        if cross[i, 4] != -1:
            if cross[i, 1] != -1:
                cross_link[i * 12 + 9, :] = [s4, v1, s1, 1]
            if cross[i, 2] != -1:
                cross_link[i * 12 + 10, :] = [s4, v1, s2, 0]
            if cross[i, 3] != -1:
                cross_link[i * 12 + 11, :] = [s4, v1, s3, 2]

    return cross_link


def Seek(car, road, cross):
    """
    规划所有车辆路径

    return : 所有车辆路径，每一行为一辆车的路径
    """

    # 构建有向图graph
    # graph每一行代表一条边，分别储存 [道路的起始路口id， 边的终点路口id， 道路长度， 道路id， graph_id, 道路限速]
    # 其中graph_id为每一条边的标识，因为双向道路占两条边，用道路id无法定位具体是哪一条边
    road_back = road[road[:, -1] == 1, :]  # 考虑双向行驶车道
    g_limit = np.hstack((road[:, 2], road_back[:, 2]))  # 记录道路的限速
    g_limit = g_limit.reshape((len(g_limit), 1))
    graph = np.stack((road[:, -3], road[:, -2], road[:, 1], road[:, 0])).transpose((1, 0))  # start, end ,length, road_id
    graph_back = np.stack((road_back[:, -2], road_back[:, -3], road_back[:, 1], road_back[:, 0])).transpose((1, 0))
    graph = np.vstack((graph, graph_back))  # start, end, length, road_id
    graph_id = np.arange(graph.shape[0]).reshape((-1, 1))
    graph = np.hstack((graph, graph_id))  # start, end, length, road_id, graph_id
    graph = np.hstack((graph, g_limit))  # start, end, length, road_id, graph_id, speed_limit

    # 计算相邻的三个路口的关系，只在dijkstra算法中用到
    cross_link = caculate_cross_link(cross, graph)  # cross_link: [s, v1, v2, direction], 符号对应dijkstra函数中表示路口的符号


    length = graph[:, 2]  # 道路长度
    num_channel = np.concatenate((road[:, 3], road_back[:, 3]))  # 道路车道数
    carRoute = np.zeros((car.shape[0], road.shape[0] + 2)).astype(int)  # 车辆规划轨迹
    heat_map = np.zeros((num_heatmap, graph.shape[0]))  # 道路拥挤程度

    for carNum in range(car.shape[0]):

        # 每10000辆车重新规划轨迹
        if carNum % 10000 == 0:
            carRoute, heat_map = reseek(carRoute, carNum, car, graph, heat_map, cross_link)

        # 获取当前车辆的发车时间
        time_slice = car[carNum, 4]  # start time

        # 更新graph的每条边的cost
        cost_heat = heat_map[time_slice, :]
        car_limit = np.ones((graph.shape[0], 2))
        car_limit[:, 0] *= car[carNum, 3]
        car_limit[:, 1] = graph[:, 5]
        speed_actual = car_limit.min(1)
        cost_length = np.ceil(length / speed_actual)
        cost_channel = np.exp(num_channel * w_channel_2)
        cost = cost_length + w_channel_1 * cost_channel + w_roadheat * cost_heat
        cost = np.clip(cost, 0.5, 10000000)

        edges = np.zeros((graph.shape[0], 3))
        edges[:, 0:2] = graph[:, 0:2]
        edges[:, 2] = cost

        # 搜索路径
        result = dijkstra(edges, cross_link, str(car[carNum, 1]), str(car[carNum, 2]))
        # 对路径进行变换
        sumarize = []
        while result[1] != ():
            sumarize.append(int(result[0]))
            if result[1] != ():
                result = result[1]
        sumarize.append(int(result[0]))  # cost, end, ..., start
        lengthSumarize = len(sumarize)

        # 车辆id和发车时间
        carRoute[carNum, 0] = car[carNum, 0]  # car id
        carRoute[carNum, 1] = time_slice  # start time
        # 将用路口表示的路径转为用road_id表示
        cost_each_route = np.zeros(lengthSumarize - 2)
        route_each_route = np.zeros(lengthSumarize - 2)
        for i in range(1, lengthSumarize - 1):
            roadchoosen = graph[graph[:, 0] == sumarize[lengthSumarize - i]]  # find the start
            roadchoosen = roadchoosen[roadchoosen[:, 1] == sumarize[lengthSumarize - i - 1]]  # find the end
            carRoute[carNum, i + 1] = roadchoosen[0, 3]
            cost_each_route[i - 1] = cost_length[roadchoosen[0, 4]]
            route_each_route[i - 1] = roadchoosen[0, 4]

        # 更新道路拥挤程度
        time_start = time_slice
        for i in range(lengthSumarize - 2):
            time_end = int(time_start + cost_each_route[i])
            change_start = max((3 * time_start - time_end) // 2, 0)
            change_end = min((3 * time_end - time_start) // 2, len(heat_map))
            # heat_map[change_start:change_end, :] += road_heat[change_start:change_end, :]
            heat_map[change_start:change_end, int(route_each_route[i])] += 1
            time_start = time_end
        print(carNum)
    return carRoute


def main():

    # 比赛官方SDK中给出的文件接口
    if len(sys.argv) != 5:
        logging.info('please input args: car_path, road_path, cross_path, answerPath')
        exit(1)
    
    car_path = sys.argv[1]
    road_path = sys.argv[2]
    cross_path = sys.argv[3]
    answer_path = sys.argv[4]
    
    logging.info("car_path is %s" % (car_path))
    logging.info("road_path is %s" % (road_path))
    logging.info("cross_path is %s" % (cross_path))
    logging.info("answer_path is %s" % (answer_path))


    # 从txt中读入数据
    carData, crossData, roadData = dataProcess(car_path, cross_path, road_path)

    # 将数据从list格式转为矩阵格式，方便处理
    car = np.array(carData).astype(int)
    cross = np.array(crossData).astype(int)
    road = np.array(roadData).astype(int)
    cross[cross[:, -1] == 1, -1] = -1

    # 调整发车顺序
    car_wait = np.zeros(car.shape).astype(int)
    time = 1
    wait_line = 0
    while wait_line != car_wait.shape[0]:  # 是否全部车已经发完
        car = car[np.argsort(car[:, 4])]  # 将车辆按照预计发车时间从低到高排列
        idx = np.argwhere(car[:, 4] <= time).max() + 1
        if idx > num_car:  # 可发车辆数大于本时间片预计发车数
            car_tmp = car[0:idx, :]
            car_tmp = car_tmp[np.random.permutation(car_tmp.shape[0]), :]  
            car_tmp = car_tmp[np.argsort(-car_tmp[:, 3])]  # 按照车速从高到底，先发车速高的车
            car[0:idx, :] = car_tmp

            car_wait[wait_line: wait_line + num_car, :] = car[0:num_car, :]
            car_wait[wait_line: wait_line + num_car, 4] = time
            car = car[num_car:car.shape[0], :]
            wait_line += num_car
        else:  # 否则，将可发车辆全部发出
            car_wait[wait_line: wait_line + idx, :] = car[0:idx, :]
            car_wait[wait_line: wait_line + idx, 4] = time
            car = car[idx:car.shape[0], :]
            wait_line += idx
        time = time + 1

    car = car_wait

    carRoute = Seek(car, road, cross)  # 规划所有车辆路径

    # 将路径写入answer.txt
    with open(answer_path, 'w') as f:
        f.write('#(carId,StartTime,RoadId...)')
        f.write('\n')

        for i in range(len(carRoute)):
            f.write('(')
            l_route = np.argwhere(carRoute[i] != 0).max() + 1
            for j in range(l_route):
                f.write(str(carRoute[i][j]))
                if j != l_route - 1:
                    f.write(', ')
                else:
                    f.write(')')
            if i != len(carRoute) - 1:
                f.write('\n')
        f.close()

    print("OK")

# to write output file


if __name__ == "__main__":
    main()
