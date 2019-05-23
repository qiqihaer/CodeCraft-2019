#include <algorithm>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <time.h>

#include "TxtHandler.h"
#include "Car.h"
#include "Cross.h"
#include "Road.h"
#include "Answer.h"
#include "Scheduler.h"
#include "Graph.h"

#define _DEBUG

using namespace std;


vector<Car> car_ve;
vector<Road> road_ve; 
vector<Cross> cross_ve;
vector<Answer> answer_ve;
map<int,int> car_map;
map<int,int> road_map;
map<int,int> cross_map;
const bool debug = false;

bool dead_lock;

double w_channel_1 = 0;
double w_channel_2 = 1;
double w_roadheat = 30;  // road_heat的权重

int num_heatmap = 5000;  // 时间片总数（会有富余，多开点内存稳一点）

int num_car_between_preset;
int num_car_after_preset;

double time_tupling = 2;

int crossKeyScalar = 100000;

double hotScaler = 400;

void write_answer(const string &path, vector<vector<int>> carRoute) {
	ofstream out(path);
    out << "#(carId,StartTime,RoadId...)" << endl;
    for (auto &iter: carRoute) {
        string str = "(" + to_string(iter[0]);
        for (unsigned int i = 1; i < iter.size(); i++) {
            str += ", " + to_string(iter[i]);
        }
        str += ")";
        out << str << endl;
    }
}

// Used to convert a group of data from the vector form into the map form.
// 用来将一组vector型的数据转换成以ID为key的map数据
map<int, vector<int>> vector2map(vector<vector<int>> v, int n) {
	map<int, vector<int>> m;
	int ID;
	vector<int> vt;
	for(auto &iter : v) {
		ID = iter[n];
		vt = iter;
		m.insert(make_pair(ID, vt));
	}
	return m;
}

class Planner {
public:
	vector<vector<int>> car;
	// 数据格式：(0-id, 1-from, 2-to, 3-speed, 4-planTime, 5-priority, 6-preset)
	
	// 下一句将其转换为对应的map格式，不过跑了之后发现性能没啥区别
	// map<int, vector<int>> car_my_map;
	// 数据格式：carId -> 0-id, 1-from, 2-to, 3-speed, 4-planTime, 5-priority, 6-preset

	vector<vector<int>> cross;
	// 数据格式：(0-id, 1-roadId, 2-roadId, 3-roadId, 4-roadId)

	vector<vector<int>> road;
	// 数据格式：(0-id, 1-length, 2-speed, 3-channel, 4-from, 5-to, 6-isDuplex)

	vector<vector<int>> preRoute;
	// 数据格式：(0-carId, 1-StartTime, 2-RoadId...)
	map<int, vector<int>> preRoute_map;
	// 数据格式：carId -> 0-carId, 1-StartTime, 2-RoadId...

	//开个数组，用来存储每个时间片的发车数
	vector<int> carLaunchNum;
	// Length = 10000

	vector<vector<int>> car_preset;
	// 数据格式：(0-id, 1-from, 2-to, 3-speed, 4-planTime, 5-priority, 6-preset)
	map<int, vector<int>> car_preset_map;
	// 数据格式：carId -> 0-id, 1-from, 2-to, 3-speed, 4-planTime, 5-priority, 6-preset

	// road_net即为路网之意，把每个方向的车道作为路网的一个组成单位，即单向道路在路网里算一个，双向道路算两个
	vector<vector<int>> road_net;
	// 数据格式：(0-from, 1-to, 2-length, 3-roadId, 4-road_netId, 5-channel, 6-speed)
	
	map<int, vector<int>> road_net_map;
	// 数据格式：road_netId -> 0-from, 1-to, 2-length, 3-roadId, 4-road_netId, 5-channel, 6-speed

	// 本算法的核心要素：路网的热度图
	vector<vector<double>> heat_map;
	// 数据格式：(time_slice, route)

	// 用cross1ID跟cross2ID来查询road(vector型数据)的map
	map<int, int> map_cross2road;

	// 用cross1ID跟roadID来查询cross2(vector型数据)的map
	map<int, int> map_cross_road2cross;

	Planner() = default;

	// Constructor
	// 构造函数
	explicit Planner(const string &roadPath, const string &carPath, const string &crossPath, const string &presetAnswerPath) {
		// Read the txt files to load the data
		// 读txt文件来加载相关数据
		cross = read_from_cross(crossPath);
		road = read_from_road(roadPath);
		preRoute = read_from_preAnswer(presetAnswerPath);
		preRoute_map = vector2map(preRoute, 0);
		car = read_from_car(carPath);
		// 转成对应的map型，方便后面一些查找型的工作
		preRoute_map = vector2map(preRoute, 0);

		// Construct the cross_to_road map and map_cross_road2cross
		// 构造上面提到的2个map，就是想实现用2个key来确定一个value的效果
		// 具体做法是将两个key做一个一一映射，使其合成为一个key：key1左移N位（即乘以10^N）+key2
		// 构造此map跟用此map查询时都这么做，定义一个常量crossKeyScalar = 10^N
		for(auto &iter : road) {
			int cross1=0, cross2=0, road=0, temp=0;
			cross1 = iter[4];
			cross2 = iter[5];
			road = iter[0];
			temp = cross1 * crossKeyScalar + cross2;
			map_cross2road[temp] = road;
			temp = cross1 * crossKeyScalar + road;
			map_cross_road2cross[temp] = cross2;
			if(iter[6] == 1) {
				cross1 = iter[5];
				cross2 = iter[4];
				temp = cross1 * crossKeyScalar + cross2;
				map_cross2road[temp] = road;
				temp = cross1 * crossKeyScalar + road;
				map_cross_road2cross[temp] = cross2;
			}
		}
		
		// Initialize the numbers of cars launched at each time slice
		// 初始化每个时间片的发车数
		carLaunchNum = initCarLaunchNum();
		
		// Determine the departure time of the non-preset cars
		// 确定每辆非预置车辆的实际发车时间
		vector<vector<vector<int>>> twoVectors = adjustDepartureTime(carLaunchNum);
		car = twoVectors[0];
		car_preset = twoVectors[1];
		
		car_preset_map = vector2map(car_preset, 0);

		// Construct the road_net
		// 构建路网数据
		road_net = road_netConstruction();
		// 将vector型的路网转换为对应的map
		// Convert the road_net from the vector form into the map form
		road_net_map = vector2map(road_net, 4);

		// Initialize an empty heat_map
		// 开数组，空的heat_map
		vector<double> emptyVector(int(road_net.size()), 0);
		heat_map.insert(heat_map.begin(), num_heatmap, emptyVector);
		// Initialize the heat map utilizing the information of the preset cars
		// 利用预置车辆的信息来对heat_map进行初始化
		initHeatMap();

	}

	// Functions utilized in the sort function
	// 在排序函数中用到的自定义比较函数，cpr意思是“比较”，数字是比较vector中的第几个元素（从0开始）
	// d指基本数据类型是double（无d则认为是int），r代表降序排列
	static bool cpr1(const vector<int> &V1, const vector<int> &V2) {return (V1[1] < V2[1]);}
	static bool cpr2dr(const vector<double> &V1, const vector<double> &V2) {return (V1[2] > V2[2]);}
	static bool cpr4(const vector<int> &V1, const vector<int> &V2) {return (V1[4] < V2[4]);}
	static bool cpr5r(const vector<int> &V1, const vector<int> &V2) {return (V1[5] > V2[5]);}
	static bool cpr6r(const vector<int> &V1, const vector<int> &V2) {return (V1[6] > V2[6]);}
	static bool cpr0d(const vector<double> &V1, const vector<double> &V2) {return (V1[0] < V2[0]);}

	// The classic dijkstra algorithm to obtain the shortest path
	// 经典dijkstra算法（迪杰斯特拉），获取最短路径
	// Inputs: 1. starting point; 2. destination; 3. ; 4. carId
	// Outputs: The shortest path consisting of a series of crossIDs
	vector<int> dijkstra(int from, int to, vector<vector<double>> g, int carId) {
		vector<int> result;
		vector<double> temp_double(2);
		vector<double> path;
		int v1, v2;
		double c, prev, next, cost;

		// 相关变量初始化
		vector<vector<double>> q;
		temp_double[0] = 0;
		temp_double[1] = from;
		q.push_back(temp_double);

		set<int> seen;				// Points that have been seen but not been determined

		map<int, double> mins;		// point<int> --> cost<double>
		mins[from] = 0;

		// cout << "Initialization accomplished!" << endl;
		
		// Initialization accomplished
		while(q.size() > 0) {
			sort(q.begin(), q.end(), cpr0d);
			path = q[0];
			cost = path[0];
			v1 = int(path[1]);
			path.erase(path.begin(), path.begin()+2);
			q.erase(q.begin());

			if(seen.find(v1) == seen.end()) {
				// cout << "v1 not in seen" << endl;
				seen.insert(v1);
				// path.insert(path.begin(), v1);
				path.push_back(v1);
				
				// 如果搜索达到目的地，则返回当前获得的路径（即为最短路径）
				if(v1 == to) {
					for(auto &iter : path) {
						result.push_back(int(iter));
					}
					return result;
				}
				// cout << "v1 != to" << endl;
				for(auto &iter : g) {
					if(iter[0] != v1) continue;
					v2 = int(iter[1]);
					if(seen.find(v2) != seen.end()) continue;
					c = iter[2];
					if(mins.count(v2) > 0){
						prev = mins[v2];
					}
					else{
						prev = -1;
					}
					next = cost + c;
					if(prev < 0 || next < prev) {
						mins[v2] = next;
						temp_double.clear();
						temp_double.push_back(next);
						temp_double.push_back(v2);
						temp_double.insert(temp_double.end(), path.begin(), path.end());
						q.push_back(temp_double);
					}
				}
			}
		}
		// 如果程序运行到这里，说明没有搜索到最短路径，即运行出了问题（理论上讲不应该出现这种情况）
		cout << "Destination not reached! from " << from << " to " << to << endl;
		for(auto &iter : path) {
			cout << iter << " -> ";
		}
		exit(1);
		path.clear();
		return result;
	}

	// To seek shortest paths for all cars
	// 给所有车辆搜索最短路径
	// No inputs
	// Outputs: carRoutes
	vector<vector<int>> seekShortestRoute(void) {
		#ifdef _DEBUG
			cout << "seekShortestRoute..." << endl;
		#endif
		vector<vector<int>> carRoutes;
		vector<vector<double>> edges;
		vector<double> edge(3,0);
		vector<int> cost_length, carRoute, result;
		vector<double> cost_heat;
		int i=0, carNum=0, time_slice=0, speed_actual=0, cl, car_id = 0, road_id=0, crossKey=0, start=0, car_speed_limit=0, f=0, t=0;
		double cost_each=0.0;
		int total = int(car.size());
		for(carNum=0; carNum<total; carNum++) {
			// 数据准备
			time_slice = car[carNum][4];	// startTime
			car_speed_limit = car[carNum][3];
			car_id = car[carNum][0];
			// if(carNum % 10000 == 0) {
			// 	carRoute = reSeekShortestRoutes(carRoute, carNum, time_slice);
			// }
			cost_heat = heat_map[time_slice];
			for(i=0; i<int(road_net.size()); i++) {
				speed_actual = min(car_speed_limit, road_net[i][6]);
				cl = road_net[i][2] / speed_actual;
				if(road_net[i][2] % speed_actual != 0) {
					cl++;
				}
				cost_length.push_back(cl);
				cost_each = min(max(cl + w_roadheat * cost_heat[i] / road_net[i][5], 0.5), 10000000.0);
				edge[0] = road_net[i][0];
				edge[1] = road_net[i][1];
				edge[2] = cost_each;
				edges.push_back(edge);
			}

			// 搜索最短路径
			result = dijkstra(car[carNum][1], car[carNum][2], edges, car_id);
			// 获得的路径是以沿途的crossID来表示的
			edges.clear();
			// cross_link

			if(result.size() == 0) {
				f = car[carNum][1];
				t = car[carNum][2];
				cout << "The result is empty! carId = " << car_id << ", from = " << f << ", to = " << t << endl;
				exit(1);
			}
			carRoute.push_back(car_id);		// Record the car_id
			carRoute.push_back(time_slice);	// Record the real startTime
			// 将沿途的crossID转化为沿途的roadID
			for(i=0; i<int(result.size()-1); i++) {
				crossKey = int(result[i]) * crossKeyScalar + int(result[i+1]);
				auto the_road = map_cross2road.find(crossKey);
				/* if(the_road == map_cross2road.end()) {
					write_cost_length("../3-map-training-1/result.txt", result);
					cout << "These two crosses are not accessable directly!" << endl;
					int cross1 = result[i];
					int cross2 = result[i+1];
					cout << "cross 1 = " << cross1 << "; cross 2 = " << cross2 << endl;
					exit(1);
				} */
				road_id = the_road->second;
				carRoute.push_back(road_id);
			}
			start = result[0];
			carRoutes.push_back(carRoute);
			// write_cost_length("../3-map-training-1/carRoute.txt", carRoute);
			// exit(1);

			// 每规划一条车的路径（包含发车时间），则对heat_map进行一次更新
			updateHeatMap(carRoute, start, time_slice, cost_length);
			cost_length.clear();
			carRoute.clear();
			#ifdef _DEBUG
				if(carNum % 10000 == 0){
					cout << "Car " << carNum << " in " << total << endl;
				}
			#endif
		}

		return carRoutes;
	}

	// To find 10% of the preset cars which are faced with the most crowded traffic in the first planing, and replan their routes
	// 找出在发车时刻整个路网中需要途径路线最堵（热度最高）的前10%的预置车辆，对它们的途径路线进行重新规划
	// 前半段是找出这10%的车，后半段基本跟seekShortestRoute是一样的，即用不一样的信息对空heat_map进行初始化之后再对所有非预置车规划一次路径
	// Inputs: carRoutes (old)
	// Outputs: carRoutes (new)
	vector<vector<int>> replanTop10(vector<vector<int>> carRoutes) {
		#ifdef _DEBUG
			cout << "replanTop10..." << endl;
		#endif
		vector<vector<double>> edges;
		vector<vector<int>> newCarRoute;
		vector<int> pre_car, route, result, cost_length, carRoute;
		int num_pre = int(preRoute.size()), carNum, f, t, i, j, pre_id, time_slice, start, road_id, route_id, route_limit, car_limit;
		int speed_actual, length, cl, top10, crossKey;
		double cost, cost_each, cost_heat_each;
		vector<vector<double>> pre_heat;
		vector<double> cost_heat, temp_double(2,0), edge(3,0);
		for(i=0; i<num_pre; i++) {
			pre_id = preRoute[i][0];
			cost = 0;
			time_slice = preRoute[i][1];
			// pre_car.clear();
			// for(auto &iter : car_preset) {
			// 	if(iter[0] == pre_id) {
			// 		pre_car = iter;
			// 		break;
			// 	}
			// }
			// if(pre_car.size() == 0) {
			// 	cout << "pre_car not found!" << endl;
			// 	exit(1);
			// }
			if(car_preset_map.count(pre_id) > 0) {
				pre_car = car_preset_map[pre_id];
			}
			else {
				cout << "pre_car not found!" << endl;
				exit(1);
			}
			
			start = pre_car[1];
			for(j=2; j<int(preRoute[i].size()); j++) {
				road_id = preRoute[i][j];
				route.clear();
				for(auto &iter : road_net) {
					if(iter[0] == start && iter[3] == road_id) {
						route = iter;
						break;
					}
				}
				if(route.size() == 0) {
					cout << "route not found!" << endl;
					exit(1);
				}
				route_id = route[4];
				start = route[1];
				route_limit = route[6];
				cost_heat_each = heat_map[time_slice][route_id];
				car_limit = pre_car[3];
				speed_actual = min(car_limit, route_limit);
				length = route[2];
				cl = length / speed_actual;
				if(length % speed_actual != 0) {
					cl++;
				}
				cost_each = min(max(cl + w_roadheat * cost_heat_each / route[5], 0.5), 10000000.0);
				cost += cost_each;
			}
			temp_double[0] = pre_id;
			temp_double[1] = cost;
			pre_heat.push_back(temp_double);
		}

		sort(pre_heat.begin(), pre_heat.end(), cpr2dr);
		top10 = int(pre_heat.size()) / 10;
		pre_heat.erase(pre_heat.begin()+top10, pre_heat.end());

		int total = int(pre_heat.size());
		for(carNum=0; carNum<total; carNum++) {
			// pre_car.clear();
			// for(auto &iter : car_preset) {
			// 	if(iter[0] == int(pre_heat[carNum][0])) {
			// 		pre_car = iter;
			// 		break;
			// 	}
			// }
			// if(pre_car.size()==0) {
			// 	cout << "pre_car not found in replanning..." << endl;
			// 	exit(1);
			// }
			if(car_preset_map.count(pre_heat[carNum][0]) > 0) {
				pre_car = car_preset_map[pre_heat[carNum][0]];
			}
			else {
				cout << "pre_car not found!" << endl;
				exit(1);
			}

			time_slice = pre_car[4];	// startTime
			car_limit = pre_car[3];
			pre_id = pre_car[0];
			cost_heat = heat_map[time_slice];
			for(i=0; i<int(road_net.size()); i++) {
				speed_actual = min(car_limit, road_net[i][6]);
				cl = road_net[i][2] / speed_actual;
				if(road_net[i][2] % speed_actual != 0) {
					cl++;
				}
				cost_length.push_back(cl);
				cost_each = min(max(cl + w_roadheat * cost_heat[i] / road_net[i][5], 0.5), 10000000.0);
				edge[0] = road_net[i][0];
				edge[1] = road_net[i][1];
				edge[2] = cost_each;
				edges.push_back(edge);
			}

			result = dijkstra(pre_car[1], pre_car[2], edges, pre_id);
			edges.clear();
			// cross_link

			if(result.size() == 0) {
				f = car[carNum][1];
				t = car[carNum][2];
				cout << "The result is empty! preId = " << pre_id << ", from = " << f << ", to = " << t << endl;
				exit(1);
			}
			carRoute.push_back(pre_id);		// Record the pre_id
			carRoute.push_back(time_slice);	// Record the real startTime
			for(i=0; i<int(result.size()-1); i++) {
				crossKey = int(result[i]) * crossKeyScalar + int(result[i+1]);
				if(map_cross2road.count(crossKey)==0) {
					write_cost_length("../3-map-training-1/result.txt", result);
					cout << "These two crosses are not accessable directly!" << endl;
					int cross1 = result[i];
					int cross2 = result[i+1];
					cout << "cross 1 = " << cross1 << "; cross 2 = " << cross2 << endl;
					exit(1);
				}
				road_id = map_cross2road[crossKey];
				carRoute.push_back(road_id);
			}
			start = result[0];
			newCarRoute.push_back(carRoute);
			// write_cost_length("../3-map-training-1/carRoute.txt", carRoute);
			// exit(1);

			updateHeatMap(carRoute, start, time_slice, cost_length);
			cost_length.clear();
			carRoute.clear();
			#ifdef _DEBUG
				if(carNum % 10000 == 0){
					cout << "Car " << carNum << " in " << total << endl;
				}
			#endif
		}

		carRoutes.insert(carRoutes.end(), newCarRoute.begin(), newCarRoute.end());
		return carRoutes;
	}

	// Construct the road_net
	// 构造路网
	// No inputs
	// Outputs: road_net
	vector<vector<int>> road_netConstruction(void) {
		#ifdef _DEBUG
			cout << "road_netConstruction..." << endl;
		#endif
		vector<vector<int>> road_net_t = road;
		for(auto &iter : road) {
			if(iter[6] == 1) {
				vector<int> temp = iter;
				temp[4] = iter[5];
				temp[5] = iter[4];
				road_net_t.push_back(temp);
			}
		}
		vector<vector<int>> road_net;
		int i = 0;
		for(auto &iter : road_net_t) {
			vector<int> temp;
			temp.push_back(iter[4]);	// from
			temp.push_back(iter[5]);	// to
			temp.push_back(iter[1]);	// length
			temp.push_back(iter[0]);	// roadId
			temp.push_back(i);			// road_netId
			temp.push_back(iter[3]);	// channel
			temp.push_back(iter[2]);	// speed
			i++;
			road_net.push_back(temp);
		}

		// write_road_net("../3-map-training-1/road_net.txt", road_net);

		return road_net;
	}

	// Initialize the numbers of cars launched at each time slice
	// 初始化每个时间片的发车数
	// No inputs
	// Outputs: carLaunchNum
	vector<int> initCarLaunchNum(void) {
		#ifdef _DEBUG
			cout << "initCarLaunchNum..." << endl;
		#endif
        int i=0;
        vector<vector<int>> route = preRoute;
        
        sort(route.begin(), route.end(), cpr1);
        // Sort preRoute according to startTime
		// 按照发车时间对预置路线进行排序

        vector<int> carLaunchNum(10000, 0);
        int preset_last = route.back()[1];
        // cout << "preset_last = " << preset_last << endl;
        for (i=0; i<preset_last; i++){
        	carLaunchNum[i] += num_car_between_preset;
        }
        for (i=preset_last; i<int(carLaunchNum.size()); i++) {
        	carLaunchNum[i] += num_car_after_preset;
        }

        return carLaunchNum;
	}

	// Determine the departure time of the non-preset cars
	// 确定每辆非预置车辆的实际发车时间
	// Inputs: carLaunchNum
	// Outputs: twoVectors -- 1st vector: car (with all the non-preset cars' departure time determined; 2nd vector: car_preset
	vector<vector<vector<int>>> adjustDepartureTime(vector<int> carLaunchNum) {
		#ifdef _DEBUG
			cout << "adjustDepartureTime..." << endl;
		#endif
		vector<vector<vector<int>>> twoVectors;
		// twoVectors[0]: car_wait; twoVectors[1]: car_preset
		vector<vector<int>> car_wait;
		vector<vector<int>> car_preset, car_preset_result;
		vector<vector<int>> car_adjust = car;
		int idx_preset = 0, i;

		for(i=0; i<int(car_adjust.size()); i++) {
			if(car_adjust[i][6] == 1) {
				idx_preset++;
				car_preset.push_back(car_adjust[i]);
			}
			else
			{
				car_wait.push_back(car_adjust[i]);
			}
		}

		if(idx_preset == 0) {
			cout << "There is no preset car!" << endl;
			exit(1);
		}
		car_adjust = car_wait;
		// the data in car_wait is meaningless at this time

		car_preset_result = car_preset;
		sort(car_preset.begin(), car_preset.end(), cpr4);

		int wait_line = 0, time = 1, num_car;

		int total = int(car_wait.size());

		while(wait_line != total) {
			// num_car: number of cars launch at this time
            num_car = carLaunchNum[time];
            // launch the preset cars
            idx_preset = 0;
            for(auto &iter : car_preset) {
            	if(iter[4] == time) {
            		idx_preset++;
            	}
            }
            if(idx_preset > 0) {
            	car_preset.erase(car_preset.begin(), car_preset.begin()+idx_preset);
            }
            // check whether the number of preset cars are less than num_car
            if(idx_preset < num_car) {
            	// sort(car_preset.begin(), car_preset.end(), cpr4);
            	sort(car_adjust.begin(), car_adjust.end(), cpr4);
				// plan_time low to high
            	int idx = 0;
            	for(auto &iter : car_adjust) {
            		if(iter[4] <= time) {
            			idx++;
            		}
            	}
            	int num_car_tmp = num_car - idx_preset;
            	// check whether cars that can be launched are more than num_car_tmp
            	if(idx > num_car_tmp) {
            		vector<vector<int>> car_tmp (car_adjust.begin(), car_adjust.begin()+idx);
            		sort(car_tmp.begin(), car_tmp.end(), cpr5r);
            		car_adjust.erase(car_adjust.begin(), car_adjust.begin()+idx);
            		car_adjust.insert(car_adjust.begin(), car_tmp.begin(), car_tmp.end());
            		car_wait.erase(car_wait.begin()+wait_line, car_wait.begin()+wait_line+num_car_tmp);
            		car_wait.insert(car_wait.begin()+wait_line, car_adjust.begin(), car_adjust.begin()+num_car_tmp);
            		for(int i=wait_line; i<wait_line+num_car_tmp; i++) {
            			car_wait[i][4] = time;
            		}
            		car_adjust.erase(car_adjust.begin(), car_adjust.begin()+num_car_tmp);
            		wait_line += num_car_tmp;
            	}
            	else {
            		car_wait.erase(car_wait.begin()+wait_line, car_wait.begin()+wait_line+idx);
            		car_wait.insert(car_wait.begin()+wait_line, car_adjust.begin(), car_adjust.begin()+idx);
            		for(int i=wait_line; i<wait_line+idx; i++) {
            			car_wait[i][4] = time;
            		}
            		car_adjust.erase(car_adjust.begin(), car_adjust.begin()+idx);
            		wait_line += idx;
            	}
            	// #ifdef _DEBUG
            	// 	cout << "Car " << wait_line << " of " << total << endl;
            	// #endif
            }
            time++;
		}

		twoVectors.push_back(car_wait);
		twoVectors.push_back(car_preset_result);
		#ifdef _DEBUG
			cout << "adjustDepartureTime accomplished!" << endl;
		#endif
		return twoVectors;
	}

	// Update the heat map each time a car's route is planned
	// 用于更新heat_map，默认每辆车从出发开始经过约200个时间片即可走完全程，在这200个时间片内，其沿途经过的路热度全部上升1
	// Inputs: 1. route; 2. starting point in the route; 3. departure time; 4. a vector encompassing the cost of each road in the whole route
	// No outputs (the member variable "heat_map" is updated)
	void updateHeatMap(vector<int> route, int start, int time_slice, vector<int> cost_length) {
		int car_id = route[0];
		int startTime = route[1];
		route.erase(route.begin(),route.begin()+2);
		vector<double> cost_each_route;
		vector<int> id_each_route;
		vector<int> roadchoosen;
		int i, j;
		for(i=0; i<int(route.size()); i++) {
			for(auto &iter : road_net) {
				if(iter[3] == route[i] && iter[0] == start) {
					roadchoosen = iter;
					break;
				}
			}
			cost_each_route.push_back(cost_length[roadchoosen[4]]*time_tupling);
			id_each_route.push_back(roadchoosen[4]);
			start = roadchoosen[1];
			roadchoosen.clear();
		}
		for(i=0; i<int(route.size()); i++) {
			for(j=time_slice; j<time_slice+200; j++) {
				heat_map[j][id_each_route[i]] += (1.0 / road_net[id_each_route[i]][5] / road_net[id_each_route[i]][2]);
			}
		}
	}

	// Initialize the heat_map utilizing the information of the preset cars
	// 利用预置车辆的信息来对heat_map进行初始化
	// No inputs
	// No outputs (the member variable "heat_map" is updated)
	void initHeatMap(void) {
		#ifdef _DEBUG
			cout << "initHeatMap..." << endl;
		#endif
		int i, time_slice=0, car_id=0, start=0, car_speed_limit;
		int cl=0, speed_actual=0;
		vector<int> route;
		vector<int> cost_length;
		// cout << "Initialize heat_map according to presetRoute..." << endl;
		for(i=0; i<int(preRoute.size()); i++) {
			route = preRoute[i];
			time_slice = route[1];
			car_id = route[0];
			auto car_data = car_preset_map.find(car_id);
			if(car_data != car_preset_map.end()) {
				start = car_data->second[1];
				car_speed_limit = car_data->second[3];
			}
			else {
				cout << "Not found! carId = " << car_id << endl;
				exit(1);
			}

			for(auto &iter : road_net) {
				speed_actual = min(car_speed_limit, iter[6]);
				cl = iter[2] / speed_actual;
				if(iter[2] % speed_actual != 0) {
					cl++;
				}
				cost_length.push_back(cl);
			}
			updateHeatMap(route, start, time_slice, cost_length);
			// cout << "Car " << i << " accomplished..." << endl;
			cost_length.clear();
		}
		// cout << "heat_map initilized!!!" << endl;
	}

	// Load the information in car.txt
	vector<vector<int>> read_from_car(const string &path) {
		#ifdef _DEBUG
			cout << "read_from_car..." << endl;
		#endif
		ifstream in(path);
	    string line;
	    getline(in, line);
	    vector<vector<int>> mapCar;
	    while (getline(in, line)) {
	        line = line.substr(1, line.size() - 2);
	        stringstream ss(line);
	        string str;
	        vector<int> intArray;
	        vector<int> car;
	        while (getline(ss, str, ','))intArray.push_back(stoi(str));
	        if(intArray[6] == 1) {
	        	// for(auto &iter : preRoute) {
	        	// 	if(iter[0] == intArray[0]){
	        	// 		intArray[4] = iter[1];
	        	// 		break;
	        	// 	}
	        	// }
	        	auto pre = preRoute_map.find(intArray[0]);
	        	if(pre != preRoute_map.end()) {
	        		intArray[4] = pre->second[1];
	        	}
	        	else {
	        		cout << "Corresponding car not found!" << endl;
	        		exit(1);
	        	}
	        }
	        car.push_back(intArray[0]); // id
	        car.push_back(intArray[1]); // from
	        car.push_back(intArray[2]); // to
	        car.push_back(intArray[3]); // speed
	        car.push_back(intArray[4]); // planTime
	        car.push_back(intArray[5]); // priority
	        car.push_back(intArray[6]); // preset

	        mapCar.push_back(car);
	    }
	    in.close();
	    return mapCar;
	}

	// Load the information in cross.txt
	vector<vector<int>> read_from_cross(const string &path) {
		#ifdef _DEBUG
			cout << "read_from_cross..." << endl;
		#endif
		ifstream in(path);
	    string line;
	    getline(in, line);
	    vector<vector<int>> mapCross;
	    while (getline(in, line)) {
	        line = line.substr(1, line.size() - 2);
	        stringstream ss(line);
	        string str;
	        vector<int> intArray;
	        vector<int> cross;
	        while (getline(ss, str, ','))intArray.push_back(stoi(str));
	        cross.push_back(intArray[0]); // id
	        cross.push_back(intArray[1]); // roadId1
	        cross.push_back(intArray[2]); // roadId2
	        cross.push_back(intArray[3]); // roadId3
	        cross.push_back(intArray[4]); // roadId4

	        mapCross.push_back(cross);
	    }
	    in.close();
	    return mapCross;
	}

	// Load the information in road.txt
	vector<vector<int>> read_from_road(const string &path) {
		#ifdef _DEBUG
			cout << "read_from_road..." << endl;
		#endif
		ifstream in(path);
	    string line;
	    getline(in, line);
	    vector<vector<int>> mapRoad;
	    while (getline(in, line)) {
	        line = line.substr(1, line.size() - 2);
	        stringstream ss(line);
	        string str;
	        vector<int> intArray;
	        vector<int> road;
	        while (getline(ss, str, ','))intArray.push_back(stoi(str));
	        road.push_back(intArray[0]); // id
	        road.push_back(intArray[1]); // length
	        road.push_back(intArray[2]); // speed
	        road.push_back(intArray[3]); // channel
	        road.push_back(intArray[4]); // from
	        road.push_back(intArray[5]); // to
	        road.push_back(intArray[6]); // isDuplex

	        mapRoad.push_back(road);
	    }
	    in.close();
	    return mapRoad;
	}

	// Load the information in preAnswer.txt
	vector<vector<int>> read_from_preAnswer(const string &path) {
		#ifdef _DEBUG
			cout << "read_from_preAnswer..." << endl;
		#endif
		ifstream in(path);
	    string line;
	    getline(in, line);
	    vector<vector<int>> preRoute;
	    while (getline(in, line)) {
	        line = line.substr(1, line.size() - 2);
	        stringstream ss(line);
	        string str;
	        vector<int> intArray;
	        vector<int> pre;
	        while (getline(ss, str, ','))intArray.push_back(stoi(str));
	        pre.push_back(intArray[0]); // carId
	        pre.push_back(intArray[1]); // startTime
	        for (unsigned int i = 2; i < intArray.size(); i++) {
	        	pre.push_back(intArray[i]); // roadId...
	        }
	        preRoute.push_back(pre);
	    }
	    in.close();
	    return preRoute;
	}
};

// main function
int main(int argc, char *argv[])
{
	// Definitions of several variables to record the running time
	clock_t begin_time, end_time, begin_time_1, end_time_1, begin_time_2, end_time_2;
	double duration, duration_1, duration_2;
	// The 3 variables without suffix are used to record the total time
	// The 3 variables with suffix 1 are used to record the time consumed for planning routes in each iteration
	// The 3 variables with suffix 2 are used to record the time consumed for checking the routes (utilizing the open source scheduler) in each iteration

	begin_time = clock();

    std::cout << "Begin" << std::endl;
	
	if(argc < 6){
		std::cout << "please input args: carPath, roadPath, crossPath, answerPath" << std::endl;
		exit(1);
	}
	
	std::string carPath(argv[1]);
	std::string roadPath(argv[2]);
	std::string crossPath(argv[3]);
	std::string presetAnswerPath(argv[4]);
	std::string answerPath(argv[5]);
	
	std::cout << "carPath is " << carPath << std::endl;
	std::cout << "roadPath is " << roadPath << std::endl;
	std::cout << "crossPath is " << crossPath << std::endl;
	std::cout << "presetAnswerPath is " << presetAnswerPath << std::endl;
	std::cout << "answerPath is " << answerPath << std::endl;

	// Definitions of 3 variables to record the planned routes for all cars
	// 定义了3个变量，用于记录所有车辆的规划轨迹
	// In each iteration, if a new workable result is obtained, the 3 variables is updated progressively
	// 每经过一次迭代，三个变量依次更新，效果如下：
	// Consequently:
	// carRoute3[k] = carRoute2[k-1] = carRoute1[k-2]
	// carRoute2[k] = carRotue1[k-1]
	// carRoute1[k] = newest result
	// "k" indicates the current iteration, "k-1" indicates the previous iteration...
	// k代表迭代的轮数（当前轮，最新的一轮），k-1指上一轮，k-2指上上轮
	vector<vector<int>> carRoute1, carRoute2, carRoute3;
	int numMax = 0;
	bool Round1 = 0;

    num_car_between_preset = 50;
	num_car_after_preset = num_car_between_preset;

	// Iteration beginning
    int round = 1;
    while(1) {
    	// cout << "Iteration: " << round++ << endl;
		// cout << "num_car = " << num_car_between_preset << endl;
    	#ifdef _DEBUG
    		cout << "Iteration: " << round++ << endl;
			cout << "num_car = " << num_car_between_preset << endl;
			cout << "Planner working..." << endl;
		#endif

		begin_time_1 = clock();

		// Initialize the planner
		// 初始化一个planner对象
		Planner planner = Planner(roadPath, carPath, crossPath, presetAnswerPath);
		// Plan routes for all cars
		// 规划所有车的路线
		carRoute1 = planner.seekShortestRoute();
		// Output the preliminary result
		// 输出此轮迭代的初步结果
		write_answer(answerPath, carRoute1);
		
		end_time_1 = clock();
		duration_1 = double(end_time_1 - begin_time_1) / CLOCKS_PER_SEC;
		#ifdef _DEBUG
			cout << "Planning time: " << duration_1 << "s" << endl;
			cout << "Scheduler working..." << endl;
		#endif

		begin_time_2 = clock();
		// Initialize the scheduler
		// 初始化调度器（判题器）
		TxtHandler th;
		th.getCarFromTxt(car_ve,carPath);
		th.getRoadFromTxt(road_ve,roadPath);
		th.getCrossFromTxt(cross_ve,crossPath);
		th.getAnswerFromTxt(answer_ve,presetAnswerPath);
		th.getAnswerFromTxt(answer_ve,answerPath);
		th.reMap(car_ve,road_ve,cross_ve);
		Scheduler scheduler;
		dead_lock = 0;
		// Check the preliminary result
		// 对初步结果进行检验
		try{scheduler.scheduleAllCars(answer_ve);}
		catch(int ID) {
			dead_lock = 1;
		}

		end_time_2 = clock();
		duration_2 = double(end_time_2 - begin_time_2) / CLOCKS_PER_SEC;
		#ifdef _DEBUG
		cout << "Scheduling time: " << duration_2 << "s" << endl;
		#endif

		// Update the carRouteX variables if no dead_lock occurs
		// 如果没有死锁，则更新用于记录路径的数据变量
		if(!dead_lock) {			
			numMax = num_car_between_preset;
			carRoute3 = carRoute2;
			carRoute2 = carRoute1;
		}
		
		// Increase the amount of cars launched in each time slice for next iteration
		// 发车强度提升，然后进行下一轮迭代
		num_car_between_preset += 5;
		num_car_after_preset = num_car_between_preset;
		car_ve.clear();
		road_ve.clear(); 
		cross_ve.clear();
		answer_ve.clear();

		end_time = clock();
		duration = double(end_time - begin_time) / CLOCKS_PER_SEC;
		#ifdef	_DEBUG
			cout << "Total time till now: " << duration << "s" << endl;
		#endif

		// If the time consumed is more than 13 mins, break
		// 如果运行超过13分钟则跳出循环
		if(duration > 780 || Round1) {
			break;
		}

    }

	// Output the ultimate result
    write_answer(answerPath, carRoute3);
	// carRoute3 is chosen rather than carRoute1 for conservativeness considering that the scheduler may not be absolutely right
	// 输出搜索到的发车强度倒数第3高的那组不死锁的结果
	// 这里的保守性（没有使用最高的一组）是考虑到检验发现判题器并非百分之百跟线上结果一致，说到底是求稳，成绩差一些总比死锁强

	#ifdef _DEBUG
		cout << "Answer output accomplished!" << endl;
	#endif

	end_time = clock();
	duration = double(end_time - begin_time) / CLOCKS_PER_SEC;
	#ifdef	_DEBUG
		cout << "numMax = " << numMax << endl;
		cout << "Total time: " << duration << "s" << endl;
	#endif

	return 0;
}
