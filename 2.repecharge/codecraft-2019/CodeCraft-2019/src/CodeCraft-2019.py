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
w_roadheat = 30  # 道路拥挤程度的权重

num_heatmap = 50000  # 给heatmap预先分配的空间（足够大就行）
num_car_max = 29  # number of cars that start at the same time
num_car_ratio = 5
num_car_min = 29

# 重新规划路径（复赛中未使用）
reseek_ratio = 0
w_roadheat_reseek = 10


class Graph:

    def __init__(self, roadPath, carPath, crossPath, presetAnswerPath):
        """
        :param roadPath: id,length,speed,channel,from,to,isDuplex
        :param carPath: id,from,to,speed,planTime
        :param crossPath: id,roadId,roadId,roadId,roadId
        """

        # 读入数据
        self.car, self.road, self.cross, self.presetRoute = self.dataRead(roadPath, carPath, crossPath, presetAnswerPath)
        
        # 区分地图，方便分地图设定参数
        if self.car.shape[0] == 65537:
            self.flag = 1
        else:
            self.flag = 2

        # 设定每个时间片发车量
        self.carLaunchNum = self.initCarLaunchNum()
        # 修改发车顺序
        self.car, self.car_preset = self.adjustDepatureTime()

        # 构建有向图，有向图的边的车道数和车速限制
        self.graph, self.num_channel, self.speed_limit = self.graphConstruction()

        # 构造cross_link，在dijkstra算法中使用
        self.cross_link = self.crossLinkConstrction()

        # 用预置车辆的路径初始化热度图（热度图表示道路被使用的情况）
        self.heat_map = np.zeros((num_heatmap, self.graph.shape[0]))  # [time_slice, route]
        self.initHeatMap()

    def findTop10(self):
        num_pre = len(self.presetRoute)
        pre_heat = np.zeros((num_pre, 2)).astype(int)
        # pre_heat: car_id, heat
        for k in range(num_pre):
            pre_id = self.presetRoute[k][0]
            pre_heat[k,0] = pre_id

            time_slice = self.presetRoute[k][1]
            pre_car = self.car_preset[self.car_preset[:, 0] == pre_id][0]
            start = pre_car[1]
            for i in range(2, np.argwhere(self.presetRoute[k] != 0).max()+1):
                road_id = self.presetRoute[k][i]
                route = self.graph[self.graph[:, 3] == road_id]
                route = route[route[:, 0] == start][0]
                route_id = route[4]
                start = route[1]
                route_limit = self.speed_limit[route_id]
                cost_heat = self.heat_map[time_slice, route_id]
                car_limit = pre_car[3]
                speed_actual = min(car_limit, route_limit)
                length = route[2]
                cost_length = np.ceil(length / speed_actual)
                num_channel = self.num_channel[route_id]
                cost = cost_length + w_roadheat * cost_heat / num_channel
                cost = np.clip(cost, 0.5, 10000000)

                pre_heat[k, 1] += cost

        pre_heat = pre_heat[np.argsort(-pre_heat[:, 1])]
        # Top10%
        pre_heat = pre_heat[:int(np.ceil(num_pre / 10))]

        pre_heat = pre_heat[:, 0]

        graph = self.graph
        num_channel = self.num_channel

        length = graph[:, 2]

        carRoute = np.zeros((len(pre_heat), self.road.shape[0] + 2)).astype(int)

        for i in range(len(pre_heat)):

            car_id = pre_heat[i]
            car = self.car_preset[self.car_preset[:, 0] == car_id][0]

            time_slice = car[4]  # start time
            cost_heat = self.heat_map[time_slice, :]
            car_limit = np.ones((graph.shape[0], 2))
            car_limit[:, 0] *= car[3]
            car_limit[:, 1] = self.speed_limit
            speed_actual = car_limit.min(1)
            cost_length = np.ceil(length / speed_actual)

            cost = cost_length + w_roadheat * cost_heat / num_channel
            cost = np.clip(cost, 0.5, 10000000)

            edges = np.zeros((graph.shape[0], 3))
            edges[:, 0:2] = graph[:, 0:2]
            edges[:, 2] = cost

            result = self.dijkstra(edges, str(car[1]), str(car[2]))

            sumarize = []
            while result[1] != ():
                sumarize.append(int(result[0]))
                if result[1] != ():
                    result = result[1]
            sumarize.append(int(result[0]))  # cost, end, ..., start

            lengthSumarize = len(sumarize)

            carRoute[i, 0] = car[0]  # car id
            carRoute[i, 1] = time_slice  # start time

            # choose road according to cross_id stored in "summarize", and append them into carRoute
            for j in range(1, lengthSumarize - 1):
                roadchoosen = graph[graph[:, 0] == sumarize[lengthSumarize - j]]  # find the start
                roadchoosen = roadchoosen[roadchoosen[:, 1] == sumarize[lengthSumarize - j - 1]]  # find the end
                carRoute[i, j + 1] = roadchoosen[0, 3]

            # renew the heat map
            self.updateHeatMap(carRoute[i], sumarize[lengthSumarize - 1], time_slice, cost_length)

        return carRoute

    def dataRead(self, roadPath, carPath, crossPath, presentAnswerPath):

        """
        :param roadPath: string, path of road.txt
        :param carPath: string, path of car.txt
        :param crossPath: string, path of car.txt
        :return: car, road, cross: numpy.ndarray
        """

        carData = []
        crossData = []
        roadData = []
        routeData = []
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
        with open(presentAnswerPath, 'r') as lines:
            for line in lines:
                line = line.split(',')
                if re.findall("\d+", line[0]) != []:
                    line[0] = re.findall("\d+", line[0])[0]
                if re.findall("\d+", line[-1]) != []:
                    line[-1] = re.findall("\d+", line[-1])[0]
                # for i in range(len(line)):
                #     line[i] = int(line[i].strip())
                routeData.append(line)

        carData = carData[1:]
        for i in range(len(carData)):
            for j in range(len(carData[i])):
                carData[i][j] = int(carData[i][j].strip())
        roadData = roadData[1:]
        for i in range(len(roadData)):
            for j in range(len(roadData[i])):
                roadData[i][j] = int(roadData[i][j].strip())
        crossData = crossData[1:]
        for i in range(len(crossData)):
            for j in range(len(crossData[i])):
                crossData[i][j] = int(crossData[i][j].strip())

        routeData = routeData[1:]
        presentRoute = np.zeros((len(routeData), len(roadData) + 2)).astype(int)
        for i in range(len(routeData)):
            for j in range(len(routeData[i])):
                presentRoute[i][j] = int(routeData[i][j].strip())

        car = np.array(carData).astype(int)
        road = np.array(roadData).astype(int)
        cross = np.array(crossData).astype(int)
        cross[cross[:, -1] == 1, -1] = -1

        # change the planTime of cars which appear in presetAnswer.txt into the real time in presetAnswer.txt
        for i in range(presentRoute.shape[0]):
            a = presentRoute[i][0]
            t = presentRoute[i][1]
            car[car[:, 0] == a, 4] = t
        # for i in range(presentRoute.shape[0]):
        #     a = presentRoute[i][0]
        #     t = presentRoute[i][1]
        #     if car[car[:, 0] == a, 4] != t:
        #         print('false0')
        return car, road, cross, presentRoute

    def initCarLaunchNum(self):
        if self.flag == 1:
            num_car_between_preset = 110
            num_car_after_preset = 110
        else:
            num_car_between_preset = 54
            num_car_after_preset = 54
        route = self.presetRoute[np.argsort(self.presetRoute[:, 1])]
        carLaunchNum = np.zeros(10000).astype(int)
        preset_last = route[-1][1]
        carLaunchNum[0: preset_last] += num_car_between_preset
        # carLaunchNum[300:preset_last] -= 8
        carLaunchNum[preset_last : len(carLaunchNum)] += num_car_after_preset

        return carLaunchNum

    def adjustDepatureTime(self):
        """
        car: (0:id, 1:from, 2:to, 3:speed, 4:planTime,  5:priority,  6:preset)
        :return:
        """
        car = self.car

        car = car[np.argsort(-car[:, 6])]
        idx_preset = np.argwhere(car[:, 6] == 1).max() + 1
        car_preset = car[0:idx_preset, :]
        car = car[idx_preset:car.shape[0], :]

        car_wait = np.zeros(car.shape).astype(int)  # car_wait only contains the cars which are not preset.

        car_preset = car_preset[np.argsort(car_preset[:, 4])]  # plan time low to high

        wait_line = 0
        time = 1

        while wait_line != car_wait.shape[0]:

            # num_car: number of cars launch at this time
            num_car = self.carLaunchNum[time]

            # launch the preset cars
            idx_preset = np.argwhere(car_preset[:, 4] == time)
            if len(idx_preset) != 0:
                idx_preset = idx_preset.max() + 1
                # car_wait[wait_line: wait_line + idx_preset, :] = car_preset[0: idx_preset]
                car_preset = car_preset[idx_preset: car_preset.shape[0], :]
                # wait_line += idx_preset
            else:
                idx_preset = 0

            # check whether the number of preset cars are less than num_car
            if idx_preset < num_car:
                car = car[np.argsort(car[:, 4])]  # plan time low to high
                idx = np.argwhere(car[:, 4] <= time).max() + 1
                num_car_tmp = num_car - idx_preset  # num_car_tmp is number of unpreset cars that can launch

                # check whether cars that can launch are more than num_car_tmp
                if idx > num_car_tmp:
                    car_tmp = car[0:idx, :]
                    car_tmp = car_tmp[np.random.permutation(car_tmp.shape[0]), :]  # random
                    # car_tmp = car_tmp[np.argsort(-car_tmp[:, 3])]  # speed  high to low
                    car_tmp = car_tmp[np.argsort(-car_tmp[:, 5])]  # priority high to low

                    car[0:idx, :] = car_tmp

                    car_wait[wait_line: wait_line + num_car_tmp, :] = car[0:num_car_tmp, :]
                    car_wait[wait_line: wait_line + num_car_tmp, 4] = time
                    car = car[num_car_tmp:car.shape[0], :]
                    wait_line += num_car_tmp
                else:
                    car_wait[wait_line: wait_line + idx, :] = car[0:idx, :]
                    car_wait[wait_line: wait_line + idx, 4] = time
                    car = car[idx:car.shape[0], :]
                    wait_line += idx
            time = time + 1

        car = self.car
        car = car[np.argsort(-car[:, 6])]
        idx_preset = np.argwhere(car[:, 6] == 1).max() + 1
        car_preset = car[0:idx_preset, :]

        return car_wait, car_preset

    def crossLinkConstrction(self):
        """
        :return: cross_link: direction of the cross (s, v1, v2) expressed by 3 crossed, (s, v1, v2, direction)
        """
        # road (id,length,speed,channel,from,to,isDuplex)
        # cross_link
        graph = self.graph
        cross = self.cross
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

    def graphConstruction(self):

        """
        construct the graph, speed_limit, num_channel
        :return:
        graph: start, end, length, road_id, route_id(graph id)
        speed_limit: speed limit of every route
        num_channel: number of channels of every route
        """

        road = self.road
        road_back = road[road[:, -1] == 1, :]

        graph = np.stack((road[:, -3], road[:, -2], road[:, 1], road[:, 0])).transpose(
            (1, 0))  # start, end ,length, road_id
        graph_back = np.stack((road_back[:, -2], road_back[:, -3], road_back[:, 1], road_back[:, 0])).transpose((1, 0))
        graph = np.vstack((graph, graph_back))  # start, end, length, road_id
        graph_id = np.arange(graph.shape[0]).reshape((-1, 1))
        graph = np.hstack((graph, graph_id))  # start, end, length, road_id, graph_id

        num_channel = np.concatenate((self.road[:, 3], road_back[:, 3]))

        g_limit = np.hstack((road[:, 2], road_back[:, 2]))
        # g_limit = g_limit.reshape((len(g_limit), 1))

        return graph, num_channel, g_limit

    def dijkstra(self, edges, f, t):

        """
        :param edges: cost of edges
        :param f: start cross
        :param t: end cross
        :return: total cost and path
        """

        cross_link = self.cross_link

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
                        w_direct = self.weight_direction(direction)
                    else:
                        w_direct = 0
                    next = cost + w_direct + c
                    if prev is None or next < prev:
                        mins[v2] = next
                        heappush(q, (next, v2, path))

        return float("inf")

    def seekShortestRoutes(self):

        graph = self.graph  # start, end, length, road_id, graph_id, speed_limit
        car = self.car

        num_channel = self.num_channel

        length = graph[:, 2]

        carRoute = np.zeros((self.car.shape[0], self.road.shape[0] + 2)).astype(int)

        for carNum in range(car.shape[0]):
            time_slice = car[carNum, 4]  # start time

            if carNum % 10000 == 0:
                carRoute = self.reSeekShortestRoutes(carRoute, carNum, time_slice)

            cost_heat = self.heat_map[time_slice, :]
            car_limit = np.ones((graph.shape[0], 2))
            car_limit[:, 0] *= car[carNum, 3]
            car_limit[:, 1] = self.speed_limit
            speed_actual = car_limit.min(1)
            cost_length = np.ceil(length / speed_actual)

            cost = cost_length + w_roadheat * cost_heat / num_channel
            cost = np.clip(cost, 0.5, 10000000)

            edges = np.zeros((graph.shape[0], 3))
            edges[:, 0:2] = graph[:, 0:2]
            edges[:, 2] = cost

            result = self.dijkstra(edges, str(car[carNum, 1]), str(car[carNum, 2]))

            sumarize = []
            while result[1] != ():
                sumarize.append(int(result[0]))
                if result[1] != ():
                    result = result[1]
            sumarize.append(int(result[0]))  # cost, end, ..., start

            lengthSumarize = len(sumarize)

            carRoute[carNum, 0] = car[carNum, 0]  # car id
            carRoute[carNum, 1] = time_slice  # start time


            # choose road according to cross_id stored in "summarize", and append them into carRoute
            for i in range(1, lengthSumarize - 1):
                roadchoosen = graph[graph[:, 0] == sumarize[lengthSumarize - i]]  # find the start
                roadchoosen = roadchoosen[roadchoosen[:, 1] == sumarize[lengthSumarize - i - 1]]  # find the end
                carRoute[carNum, i + 1] = roadchoosen[0, 3]

            # renew the heat map
            self.updateHeatMap(carRoute[carNum], sumarize[lengthSumarize - 1], time_slice, cost_length)

        return carRoute

    def weight_direction(self, direction):
        if direction == 0:
            return 0
        if direction == 1:
            return 10
        if direction == 2:
            return 20

    def reSeekShortestRoutes(self, carRoute, carNum, time_slice):

        graph = self.graph
        car = self.car


        reseek_num = int(round(reseek_ratio * carNum))
        reseek_index = np.random.permutation(np.arange(carNum))[0: reseek_num]

        length = graph[:, 2]

        cost_heat = self.heat_map[time_slice, :]

        for i in range(reseek_num):

            car_limit = np.ones((graph.shape[0], 2))
            car_limit[:, 0] *= car[reseek_index[i], 3]
            car_limit[:, 1] = self.speed_limit
            speed_actual = car_limit.min(1)
            cost_length = np.ceil(length / speed_actual)

            cost = cost_length + w_roadheat_reseek * cost_heat
            cost = np.clip(cost, 0.5, 10000000)

            edges = np.zeros(graph.shape)
            edges[:, 0:2] = graph[:, 0:2]
            edges[:, 2] = cost

            result = self.dijkstra(edges, str(car[reseek_index[i], 1]), str(car[reseek_index[i], 2]))

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

            s = car[reseek_index[i], 1]  # start
            for j in range(2, np.argwhere(carRoute_old != 0).max() + 1):
                roadchoosen = graph[graph[:, 3] == carRoute_old[j]]
                roadchoosen = roadchoosen[roadchoosen[:, 0] == s]
                s = roadchoosen[0, 1]
            time_slice = carRoute_old[1]

            self.updateHeatMap(carRoute_old, car[reseek_index[i], 1], time_slice, cost_length)

            # renew carRoute and road_heat
            for j in range(1, lengthSumarize - 1):
                roadchoosen = graph[graph[:, 0] == sumarize[lengthSumarize - j]]  # find the start
                roadchoosen = roadchoosen[roadchoosen[:, 1] == sumarize[lengthSumarize - j - 1]]  # find the end
                carRouteTemp[j + 1] = roadchoosen[0, 3]
            carRoute[reseek_index[i], :] = carRouteTemp

            time_slice = carRouteTemp[1]

            self.updateHeatMap(carRouteTemp, sumarize[lengthSumarize - 1], time_slice, cost_length)

        return carRoute

    def updateHeatMap(self, route, start, time_slice, cost_length):
        route = route[2:np.argwhere(route != 0).max()+1]
        cost_each_route = np.zeros(len(route))
        route_each_route = np.zeros(len(route))
        for i in range(len(route)):
            roadchoosen = self.graph[self.graph[:, 3] == route[i], :]  # road_id
            roadchoosen = roadchoosen[roadchoosen[:, 0] == start]  # start
            cost_each_route[i] = cost_length[roadchoosen[0, 4]]
            route_each_route[i] = roadchoosen[0, 4]
            start = roadchoosen[0, 1]

        cost_each_route *= time_tupling
        time_start = time_slice
        for i in range(len(route)):
            time_end = int(time_start + cost_each_route[i])
            change_start = int(max(time_start - time_extend_ratio * (time_end - time_start), time_slice))
            change_end = int(min(time_end + time_extend_ratio * (time_end - time_start), time_slice + 200))
            self.heat_map[change_start:change_end, int(route_each_route[i])] += 1
            time_start = time_end

    def initHeatMap(self):
        """
        update self.heat_map with preset route in self.presetRoute
        """
        for i in range(self.presetRoute.shape[0]):
            route = self.presetRoute[i]
            time_slice = route[1]
            car_id = route[0]

            start = self.car_preset[self.car_preset[:, 0] == car_id][0][1]
            car_speed_limit = self.car_preset[self.car_preset[:, 0] == car_id][0][3]

            length = self.graph[:, 2]
            car_limit = np.ones((self.graph.shape[0], 2))
            car_limit[:, 0] *= car_speed_limit
            car_limit[:, 1] = self.speed_limit
            speed_actual = car_limit.min(1)
            cost_length = np.ceil(length / speed_actual)

            self.updateHeatMap(route, start, time_slice, cost_length)

    def checkRoute(self, carRoute):
        for i in range(carRoute.shape[0]):
            c = self.car[i]
            start = c[1]
            end = c[2]
            route = carRoute[i]
            if route[0] != c[0]:
                print('carid: false')
                break
            if route[1] != c[4]:
                print('plantime: false')
                break
            route = route[2: np.argwhere(route != 0).max()+1]
            for j in range(len(route)):
                r = self.graph[self.graph[:, 3] == route[j]]
                r = r[r[:, 0] == start]
                start = r[0][1]
            if start != end:
                print('route: false')
                break


def main():
    if len(sys.argv) != 6:
        logging.info('please input args: car_path, road_path, cross_path, answerPath')
        exit(1)

    car_path = sys.argv[1]
    road_path = sys.argv[2]
    cross_path = sys.argv[3]
    preset_answer_path = sys.argv[4]
    answer_path = sys.argv[5]

    logging.info("car_path is %s" % (car_path))
    logging.info("road_path is %s" % (road_path))
    logging.info("cross_path is %s" % (cross_path))
    logging.info("preset_answer_path is %s" % (preset_answer_path))
    logging.info("answer_path is %s" % (answer_path))

    # 初始化地图
    graph = Graph(road_path, car_path, cross_path, preset_answer_path)

    # 寻找地图最短路径
    carRoute = graph.seekShortestRoutes()

    # 重新规划10%预置车辆的轨迹
    preTop10 = graph.findTop10()
    carRoute = np.vstack((carRoute, preTop10))

    # 将车辆轨迹的文件写入answer.txt
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


if __name__ == "__main__":
    main()
