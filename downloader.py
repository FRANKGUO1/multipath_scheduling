import os
import shutil
import time
import numpy
from ctypes import *
from util import *
from multiprocessing import Process, Manager
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
from collections import defaultdict
import threading
from typing import Optional

from mpd import parse_mpd
from bitrate import get_initial_bitrate, get_bitrate_level, get_max_bitrate, get_nb_bitrates, get_resolution
from bitrate import bitrate_mapping, build_arms
from config import playback_buffer_map, max_playback_buffer_size, rebuffering_ratio, history
from config import nb_paths, playback_buffer_frame_ratio, DEBUG_SEGMENTS, target_ip, network_interface1, network_interface2
import config
from get_delay_throughput import monitor_path

default_mpd_url = "../mpd/stream.mpd"  # 提供mpd文件的地址
default_host = "10.0.1.2"
default_port = 4443
tmp_dir = "./tmp"

DEBUG_WAIT = False
DEBUG_WAIT_SECONDS = 10
DEBUG = True


# 进度详解，目前qlearning的算法已经了解，q表设置也解决，现在需要的是q表初始化以及q表的更新和决策效果。


# 这里是通过将path_id映射成为网卡，通过http3请求发送出去。
# 这里和mininet的有冲突
if_name_mapping = {
    1: "h2-eth0",  # 高带宽高时延
    2: "h2-eth1",  # 低带宽低时延
    -1: "h2-eth0",
}

interface_mapping = {
    network_interface1: 1, 
    network_interface2: 2
}

interface_data = {
    network_interface1: {"latency": [], "throughput": []},
    network_interface2: {"latency": [], "throughput": []},
}


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def clear_download_dir():
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)


def get_playback_buffer_size() -> int:
    return len(playback_buffer_map)


class DownloadStat(Structure):
    """
    typedef struct picoquic_download_stat {
        double time;
        double throughput;
        uint64_t rtt;
        uint64_t one_way_delay_avg;
        uint64_t bandwidth_estimate;
        uint64_t total_bytes_lost;
        uint64_t total_received;
        uint64_t data_received;
    }picoquic_download_stat;
    """

    _fields_ = [
        ("time", c_double),
        ("throughput", c_double),
        ("rtt", c_uint64),
        ("one_way_delay_avg", c_uint64),
        ("bandwidth_estimate", c_uint64),
        ("total_bytes_lost", c_uint64),
        ("total_received", c_uint64),
        ("data_received", c_uint64),
    ]


class SimpleStateDiscretizer:
    def __init__(self):
        # 为每条路径定义RTT和throughput的固定范围
        # RTT范围: <10, 10-50, 50-100, >100 (4个区间)
        self.rtt_ranges = [10, 50, 100]
        
        # Throughput范围: <5, 5-10, 10-20, 20-30, >30 (5个区间)
        self.throughput_ranges = [5, 10, 20, 30]
        
        # 计算单条路径的状态数
        self.rtt_states = len(self.rtt_ranges) + 1  # 4个状态
        self.throughput_states = len(self.throughput_ranges) + 1  # 5个状态
        
        # 单条路径的状态数 = RTT状态数 × Throughput状态数 = 4 × 5 = 20
        self.single_path_states = self.rtt_states * self.throughput_states
    
    def discretize_path_state(self, rtt, throughput):
        """
        将单条路径的RTT和throughput转换为离散状态
        
        参数:
        rtt: float, RTT值(ms)
        throughput: float, 吞吐量值(Mbps)
        
        返回:
        tuple: (rtt_state, throughput_state) 离散化后的状态编号
        """
        rtt_state = numpy.digitize(rtt, self.rtt_ranges)
        throughput_state = numpy.digitize(throughput, self.throughput_ranges)
        return (rtt_state, throughput_state)
    
    def get_path_state_number(self, rtt, throughput):
        """
        将单条路径的RTT和throughput转换为单个状态编号
        
        参数:
        rtt: float, RTT值(ms)
        throughput: float, 吞吐量值(Mbps)
        
        返回:
        int: 该路径的状态编号（从0开始）
        """
        rtt_state, throughput_state = self.discretize_path_state(rtt, throughput)
        return rtt_state * self.throughput_states + throughput_state
    
    def get_multi_path_state(self, path1_rtt, path1_throughput, path2_rtt, path2_throughput):
        """
        将两条路径的状态组合为一个总体状态编号
        
        参数:
        path1_rtt: float, 路径1的RTT值(ms)
        path1_throughput: float, 路径1的吞吐量值(Mbps)
        path2_rtt: float, 路径2的RTT值(ms)
        path2_throughput: float, 路径2的吞吐量值(Mbps)
        
        返回:
        int: 组合后的状态编号（从0开始）
        """
        # 获取两条路径各自的状态编号
        path1_state = self.get_path_state_number(path1_rtt, path1_throughput)
        path2_state = self.get_path_state_number(path2_rtt, path2_throughput)
        
        # 将两个路径的状态组合成一个总状态
        # path1_state的范围是0到19，path2_state也是0到19
        # 总状态 = path1_state * 20 + path2_state
        return path1_state * self.single_path_states + path2_state
    
    def get_state_space_size(self):
        """
        返回总的状态空间大小
        
        返回:
        int: 状态空间的总大小 (400 = 20 × 20)
        """
       
        return self.single_path_states * self.single_path_states


class Downloader:
    # define an init function with optional parameters and type hints

    # default_mpd_url = "../mpd/stream.mpd"
    # default_host = "10.0.1.2"
    # default_port = 4443
    # tmp_dir = "./tmp"

    
    def __init__(self, 
                 host: str = default_host,
                 port: int = default_port,
                 mpd_url: str = default_mpd_url,
                 scheduler: str = "contextual_bandit",
                 algorithm: str = "",
                 discretizer: Optional[SimpleStateDiscretizer] = None,
                 ):
        self.host = host
        self.port = port
        self.manager = Manager()
        self.scheduler = scheduler
        self.algorithm = algorithm

        if discretizer is None:
            discretizer = SimpleStateDiscretizer()
    
        self.discretizer = discretizer
        self.num_actions = nb_paths * get_nb_bitrates()
    
        # qtable暂时在[0,1)之间随机初始化
        self.q_table = numpy.random.uniform(0, 1, (
            self.discretizer.get_state_space_size() * self.discretizer.get_state_space_size(),  # 路径的状态空间
            self.num_actions # 动作组合数
        ))

        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # ε-贪婪策略的参数

        self.alpha = 0.7
        self.beta = 0.3

        self.radius = None
        self.initial_explore_done = False
        self.scheduled = defaultdict(int) # 访问不存在的值时，默认为0且不报错

        self.download_queue = [] 
        for i in range(nb_paths):
            self.download_queue.append(self.manager.Queue(maxsize=1)) # 这里的下载队列是两个队列对象，每一个队列对象最大容量为1

        self.libplayer = CDLL("./libplayer.so") # so 文件是 Linux 下的共享库格式，该文件实现了HTTP3请求的C库，可以直接调用

        # 解析MPD，获取视频片段的字典，这个字典包含
        """
        {
            "resolution_height(1080/720/..)": {
                level_1(1/2/3): [
                    {
                        "duration_seconds": duration_value/timescale,
                        "duration": duration_value,
                        "frame_count": frame_count_value, 帧数=
                        "url": segment_url('url': 'video/avc1/2/seg-268.m4s'，就是视频片段的文件路径)
                    },
                    {
                        # 视频片段信息
                    },
                    # More segments
                ],
                level_2: [
                    {
                        "duration_seconds": duration_value,
                        "duration": duration_value,
                        "frame_count": frame_count_value,
                        "url": segment_url
                    },
                    # More segments
                ]
            },
            # More resolutions
        }

        """
        self.url = parse_mpd(mpd_url) # default_mpd_url = "../mpd/stream.mpd" 将mpd文件（目前来看是在本地）的地址传递给parse_mpd函数，获取url。
        
        # init_resolution=1080，init_bitrate_level=2，init_bitrate=71817.751
        self.init_resolution, self.init_bitrate_level, self.init_bitrate = get_initial_bitrate()

        print("downloader inited with host: %s, port: %d" % (self.host, self.port))
        print("initial bitrate level: %d, initial bitrate: %f" % (self.init_bitrate_level, self.init_bitrate))

        self.downloaded_segments = 0


    def main_loop(self):
        """
        理清程序的顺序，首先先是contextual_bandit_scheduling(),然后是initial_explore(),
        然后是
        """
        if self.scheduler == "contextual_bandit":
            scheduling_process = Process(target=self.contextual_bandit_scheduling)
        elif self.scheduler == "q_learning":
            scheduling_process = Process(target=self.q_learning_scheduling)
        elif self.scheduler == "roundrobin":
            scheduling_process = Process(target=self.roundrobin_scheduling)
        elif self.scheduler == "minrtt":
            scheduling_process = Process(target=self.minrtt_scheduling)
        else:
            # default scheduler set to roundrobin
            scheduling_process = Process(target=self.roundrobin_scheduling)

        scheduling_process.start() # 开启调度算法

        process_list = []
        for path_id in range(nb_paths):
            download_process = Process(target=self.download, args=(path_id,))
            process_list.append(download_process)
            download_process.start() # 启动下载

        scheduling_process.join() # 等待停止调度算法停止
        for path_id in range(nb_paths):
            process_list[path_id].join()


    def update_qtable(self, history):
        def get_history_action():
            # history在download函数中处理
            _history_action = []
            for x in history:
                path_id = x["path_id"]
                bitrate = x["bitrate"]
                resolution = x["resolution"]
                # 由清晰度和比特率获取比特率水平
                bitrate_level = get_bitrate_level(resolution, bitrate)
                nb_bitrates = get_nb_bitrates()
                _history_action.append((path_id - 1) * nb_bitrates + bitrate_level)
                # 和task["arm"] = (path_id - 1) * get_nb_bitrates() + k一样的计算方法

            return _history_action
        
        def get_history_state():
            _history_state = []
            for x in history:
                state_mapping = self.discretizer.get_multi_path_state(x["path1_rtt"], x["path1_throughput"], x["path2_rtt"], x["path2_throughput"])
                _history_state.append(state_mapping)
            return _history_state
        
        def get_history_quality():
            _history_quality = []
            for x in history:
                quality = (x["bitrate_ratio_before"], x["bitrate_ratio"])
                _history_quality.append(quality)
            return _history_quality
        
        def get_history_rebuffering_ratio():
            _history_rebuffering_ratio = []
            for x in history:
                rebuffering_ratio = (x["rebuffering_ratio"], x["rebuffering_ratio_after"])
                _history_rebuffering_ratio.append(rebuffering_ratio)
            return _history_rebuffering_ratio
        
        if len(history) >= 2:
            history_action = get_history_action()
            history_state = get_history_state()
            history_quality = get_history_quality()
            history_rebuffering_ratio = get_history_rebuffering_ratio()






    # 这个函数用于向服务器发请求，下载相关视频片段
    def download(self, path_id: int):
        while True:
            if self.download_queue[path_id].qsize() > 0:
                # 监听下载队列，若下载队列有任务，则获取
                task = self.download_queue[path_id].get()
                if task["eos"] == 1:
                    break

                stat = DownloadStat() # 获取rtt，throughput等指标
                
                # 在player处获得,在发起请求前，先获取重缓存率。
                _playback_buffer_ratio_before = playback_buffer_frame_ratio.value
                _rebuffering_ratio_before = rebuffering_ratio.value

                if history is None:
                    _bitrate_level_ratio_before = 0
                else:
                    _bitrate_level_ratio_before = history[-1]["bitrate_ratio"]

                # 在请求之前获取路径的状态信息
                path1_statistics = monitor_path(target_ip, network_interface1)
                path2_statistics = monitor_path(target_ip, network_interface2)

                # 这一部分就是利用libplayer库，向quic服务器发起请求，具体得看client.h，libplayer.so就是链接这个文件
                with stdout_redirected(): # 使得下载过程C库的输出直接丢弃
                    self.libplayer.download_segment(c_str(self.host), # 服务器的主机地址 
                                                    c_int(self.port), # 服务器的端口
                                                    c_str(task["url"]), # 片段的url
                                                    c_str(if_name_mapping[task["path_id"]]), # 路径id，对应网络接口
                                                    c_str(tmp_dir), # 下载的临时目录
                                                    byref(stat))  # 获取参数

                print("downloaded %s on path %s" % (task["url"], task["path_id"]))
                self.downloaded_segments += 1 # 下载片段加1

                playback_buffer_map[task["seg_no"]] = task

                _playback_buffer_ratio = playback_buffer_frame_ratio.value
                _rebuffering_ratio = rebuffering_ratio.value
                _bitrate_level_ratio = task["bitrate"] * 1.0 / get_max_bitrate()  # 用当前的比特率除以最大的比特率就是目前的比特率比例，越接近1越好。
                

                # 保存历史数据,这里来保存以往的数据
                history.append({
                    # q-learning添加部分
                    "path1_throughput": path1_statistics["throughput"],
                    "path2_throughput": path2_statistics["throughput"],
                    "path1_rtt": path1_statistics["rtt"],
                    "path2_rtt": path2_statistics["rtt"],

                    "bitrate_ratio": _bitrate_level_ratio,
                    "bitrate_ratio_before": _bitrate_level_ratio_before,
                    
                    # mab
                    "throughput": stat.throughput,
                    "rtt": stat.one_way_delay_avg / 1000.0 * 2,
                    "arm": task["arm"],
                    "seg_no": task["seg_no"],
                    "resolution": task["resolution"],
                    "bitrate": task["bitrate"],            
                    "playback_buffer_ratio": _playback_buffer_ratio_before,
                    "rebuffering_ratio": _rebuffering_ratio_before,
                    "playback_buffer_ratio_after": _playback_buffer_ratio,
                    "rebuffering_ratio_after": _rebuffering_ratio,
                    "path_id": task["path_id"],
                    "initial_explore": task["initial_explore"],
                    # 如果重缓存率下降，增大码率水平，若上升,则减小码率水平
                    "reward": -1 * _bitrate_level_ratio if _rebuffering_ratio - _rebuffering_ratio_before > 0 else _bitrate_level_ratio,      
                })

                self.update_qtable(history)

                # 在探索阶段完成后，利用历史信息来进一步优化模型
                if self.scheduler == "contextual_bandit":
                    if self.initial_explore_done:
                        self.radius.partial_fit(decisions=pd.Series([task["arm"]]),
                                                rewards=pd.Series([history[-1]["reward"]]), contexts=task["context"])

                self.download_queue[path_id].task_done()

                if DEBUG_WAIT:
                    time.sleep(DEBUG_WAIT_SECONDS)


    def sequential_scheduling(self):
        for i in range(1, len(self.url[self.init_resolution][self.init_bitrate_level])):
            task = self.url[self.init_resolution][self.init_bitrate_level][i]
            task["seg_no"] = i
            task["resolution"] = self.init_resolution
            task["bitrate"] = self.init_bitrate
            task["path_id"] = -1
            task["eos"] = 0
            self.download_queue[-1].put(task)

        self.download_queue[-1].put({
            "eos": 1
        })

    def get_latest_bw_on_path(self, path_id: int):
        for i in range(len(history) - 1, -1, -1):
            if history[i]["path_id"] == path_id:
                return history[i]["throughput"]
        return 0

    """
    :return min_rtt_path_id (0, 1)
    """
    def get_minrtt_path_id(self):
        min_rtt = 999999
        min_rtt_path_id = 0
        for path_id in range(nb_paths):
            for i in range(len(history) - 1, -1, -1):
                if history[i]["path_id"] - 1 == path_id:
                    if history[i]["rtt"] < min_rtt:
                        min_rtt = history[i]["rtt"]
                        min_rtt_path_id = path_id
                    break
        return min_rtt_path_id
    
      
    def initial_explore(self):
        # nb_paths为2
        seg_no = 1
        for path_id in range(1, nb_paths + 1):
            for r, m in reversed(bitrate_mapping.items()): # key: {key: value}
                """
                bitrate_mapping = {
                    1080: {
                        1: 87209.263,
                        2: 71817.751,
                        3: 55301.205,
                    },
                    720: {
                        4: 35902.455,
                        5: 22572.278,
                    },
                    360: {
                        6: 4481.84,
                    },
                }
                """
                for k, v in reversed(m.items()):
                    """
                    r是分辨率高度,k是比特率等级等级,seg_no=1,就是指索引为1的字典,因为索引为0的片段是初始化视频，然后顺序依此遍历视频片段
                    {'duration_seconds': 2.0, 'duration': 24576, 'frame_count': 48.0, 'url': 'video/avc1/1/seg-1.m4s'}
                    """
                    task = self.url[r][k][seg_no]
                    # 给task字典添加新的k-v
                    task["seg_no"] = seg_no 
                    task["resolution"] = r
                    task["bitrate"] = v
                    task["path_id"] = path_id
                    task["eos"] = 0 # 等于1时表示视频段播放完，为0表示未播放完
                    task["initial_explore"] = 1
                    task["arm"] = (path_id - 1) * get_nb_bitrates() + k # 记录每一个比特率等级的标识，路径1则是1-3，路径2则是6+（1-3）
                    self.scheduled[seg_no] = 1 # 表示分辨率和比特率等级下的片段已经调度
                    self.download_queue[path_id - 1].put(task) # 这里使用path_id作为索引，0和1索引，表示两条路径的下载任务
                    self.download_queue[path_id - 1].join() # 等待多线程任务结束
                    seg_no += 1 # 处理下一个片段

        print(f"{bcolors.RED}all initial explore tasks are put into queue {bcolors.ENDC}")
        # 表明所有初始化探索任务被放进队列中了

        for i in range(nb_paths):
            self.download_queue[i].join()  # 确保任务全部完成

        for h in history:
            print(h) # 这里history是可以进程共享的
    """
    get the highest resolution, bitrate and bitrate_level to the given bandwidth
    :return: resolution, bitrate, bitrate_level
    """
    def get_closest_resolution_and_bitrate_level(self, bw: float):
        r, b, bl = 360, 4481.84, 6
        for resolution in bitrate_mapping:
            for bitrate_level in bitrate_mapping[resolution]:
                if bitrate_mapping[resolution][bitrate_level] < bw * 1000:
                    r = resolution
                    b = bitrate_mapping[resolution][bitrate_level]
                    bl = bitrate_level
                    return r, b, bl
        return r, b, bl

    def roundrobin_scheduling(self):
        print("##roundrobin scheduling is selected##")
        i = 1
        last_path_id = 0
        while True:
            if DEBUG and i > DEBUG_SEGMENTS:
                playback_buffer_map[i] = {
                    "eos": 1,
                    "frame_count": 0,
                }
                for i in range(nb_paths):
                    task = dict()
                    task["path_id"] = i+1
                    task["eos"] = 1
                    task["initial_explore"] = 0
                    self.download_queue[i].put(task)
                break

            if playback_buffer_frame_ratio.value >= 1:
                continue

            if self.scheduled[i] == 1:
                continue

            # 0, 1
            path_id = last_path_id ^ 1
            last_path_id = path_id

            bw = self.get_latest_bw_on_path(path_id+1)

            resolution, bitrate, bitrate_level = self.get_closest_resolution_and_bitrate_level(bw)
            print("bw on path %d is %f, r: %d, b: %f, bl: %d" % (path_id+1, bw, resolution, bitrate, bitrate_level))

            seg_no = i

            task = self.url[resolution][bitrate_level][seg_no]
            task["seg_no"] = seg_no
            task["arm"] = -1
            task["resolution"] = resolution
            task["bitrate"] = bitrate
            task["path_id"] = path_id+1
            task["eos"] = 0
            task["initial_explore"] = 0
            self.scheduled[seg_no] = 1
            self.download_queue[path_id].put(task)

            i += 1

    def minrtt_scheduling(self):
        print("##minrtt scheduling is selected##")

        self.initial_explore()
        self.initial_explore_done = True

        i = 13
        while True:
            if DEBUG and i > DEBUG_SEGMENTS:
                playback_buffer_map[i] = {
                    "eos": 1,
                    "frame_count": 0,
                }
                for i in range(nb_paths):
                    task = dict()
                    task["path_id"] = i + 1
                    task["eos"] = 1
                    task["initial_explore"] = 0
                    self.download_queue[i].put(task)
                break

            if playback_buffer_frame_ratio.value >= 1:
                continue

            if self.scheduled[i] == 1:
                continue

            # 0, 1
            path_id = self.get_minrtt_path_id()
            bw = self.get_latest_bw_on_path(path_id + 1)

            resolution, bitrate, bitrate_level = self.get_closest_resolution_and_bitrate_level(bw)
            print("bw on path %d is %f, r: %d, b: %f, bl: %d" % (path_id + 1, bw, resolution, bitrate, bitrate_level))

            seg_no = i

            task = self.url[resolution][bitrate_level][seg_no]
            task["seg_no"] = seg_no
            task["arm"] = -1
            task["resolution"] = resolution
            task["bitrate"] = bitrate
            task["path_id"] = path_id + 1
            task["eos"] = 0
            task["initial_explore"] = 0
            self.scheduled[seg_no] = 1
            self.download_queue[path_id].put(task)

            i += 1


    def qlearning_initial_explore(self):
        seg_no = 1

        # 每个动作都尝试一次，共12个动作，这个作为初始动作
        for path_id in range(1, nb_paths + 1):
            for r, m in reversed(bitrate_mapping.items()):
                for k, v in reversed(m.items()):
                    # 创建下载任务
                    task = self.url[r][k][seg_no]
                    task["seg_no"] = seg_no 
                    task["resolution"] = r
                    task["bitrate"] = v
                    task["path_id"] = path_id
                    task["eos"] = 0 # 等于1时表示视频段播放完，为0表示未播放完
                    task["initial_explore"] = 1
                    task["action"] = (path_id - 1) * get_nb_bitrates() + k
         
                    self.scheduled[seg_no] = 1
                    self.download_queue[path_id - 1].put(task)
                    self.download_queue[path_id - 1].join()
                    seg_no += 1
                    
                   
        print(f"{bcolors.RED}Initial exploration completed, Q-table initialized{bcolors.ENDC}")
        
        # 等待所有任务完成
        for i in range(nb_paths):
            self.download_queue[i].join()
    
        for h in history:
            print(h)

    
    def q_learning_scheduling(self):
        def get_history_action():
            # history在download函数中处理
            _history_action = []
            for x in history:
                path_id = x["path_id"]
                bitrate = x["bitrate"]
                resolution = x["resolution"]
                # 由清晰度和比特率获取比特率水平
                bitrate_level = get_bitrate_level(resolution, bitrate)
                nb_bitrates = get_nb_bitrates()
                _history_action.append((path_id - 1) * nb_bitrates + bitrate_level)
                # 和task["arm"] = (path_id - 1) * get_nb_bitrates() + k一样的计算方法

            return _history_action
        
        def get_history_state():
            _history_state = []
            for x in history:
                state_mapping = self.discretizer.get_multi_path_state(x["path1_rtt"], x["path1_throughput"], x["path2_rtt"], x["path2_throughput"])
                _history_state.append(state_mapping)
            return _history_state
        
        def get_history_quality():
            _history_quality = []
            for x in history:
                quality = (x["bitrate_ratio_before"], x["bitrate_ratio"])
                _history_quality.append(quality)
            return _history_quality
        
        def get_history_rebuffering_ratio():
            _history_rebuffering_ratio = []
            for x in history:
                rebuffering_ratio = (x["rebuffering_ratio"], x["rebuffering_ratio_after"])
                _history_rebuffering_ratio.append(rebuffering_ratio)
            return _history_rebuffering_ratio

        def get_smallest_unscheduled_segment():
            for i in range(1, nb_segments):
                if self.scheduled[i] == 0:
                    return i

        def parse_action(action) -> (int, int, float, int):
            path_id = (action - 1) // get_nb_bitrates() + 1
            bitrate_level = (action - 1) % get_nb_bitrates() + 1

            # get resolution from bitrate_level in bitrate_mapping
            resolution = get_resolution(bitrate_level)
            return path_id, resolution, bitrate_mapping[resolution][bitrate_level], bitrate_level

        def choose_unscheduled_path_action(state):
            # 利用：选择Q值最大的动作
            action = numpy.argmax(self.q_table[state][200:]) + 200 
            return action

        def choose_action(state):
            """
            使用ε-贪婪策略选择动作
            
            参数:
            state: 离散化后的状态元组
            """
            if numpy.random.random() < self.epsilon:
                # 探索：随机选择动作
                action = numpy.random.randint(self.num_actions)
                return action
            else:
                # 利用：选择Q值最大的动作
                action = numpy.argmax(self.q_table[state]) 
                return action

        def calculate_reward(quality, rebuffering_ratio):
            """
            计算奖励值
            
            参数:
            quality: (上一个视频片段比特率水平率, 本视频片段比特率水平率)
            rebuffering_ratio: 重缓冲比率 (上一个视频片段重缓存率, 本视频片段重缓存率)
            """

            # 计算变化量
            quality_change = quality[1] - quality[0]  # 比特率水平变化范围是[-1, 1]
            
            # 对重缓存率使用对数变换后计算变化
            # 加1是为了避免log(0)的情况
            rebuffer_change = numpy.log(rebuffering_ratio[1] + 1) - numpy.log(rebuffering_ratio[0] + 1)
            
            # 重缓存率的对数变化也可以用arctan进一步归一化到[-1, 1]范围
            normalized_rebuffer_change = 2 * numpy.arctan(rebuffer_change) / numpy.pi
            
            # 设置权重
            quality_weight = 3.0
            rebuffer_weight = -1.5
            
            # 计算奖励
            reward = quality_weight * quality_change + rebuffer_weight * normalized_rebuffer_change
            
            return reward
        
        # 更新q表应该单独写一个线程，不断读取history，一旦有新的就更新
        def update(state, action, reward, next_state):
            """
            更新Q表
            
            参数:
            state: 当前状态
            action: 执行的动作 (path, bitrate)
            reward: 获得的奖励
            next_state: 下一个状态
            """
            current_q = self.q_table[state][action]
            
            # 计算下一状态的最大Q值
            next_max_q = numpy.max(self.q_table[next_state])
            
            # Q-learning更新公式
            new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * next_max_q - current_q
            )
            
            self.q_table[state][action] = new_q

        def train_step(current_state, action, next_state, quality, rebuffering_ratio):
            """
            训练一个步骤
            
            参数:
            current_paths_state: 当前路径状态
            action: 执行的动作
            quality: 获得的视频质量
            rebuffering_ratio: 重缓冲比率
            paths_state_before: 下一个路径状态
            """
            # 需要获取之前状态，则需要得知之前路径的rtt和吞吐量，就是在进行调度前的路径状态，
            # 上一个状态的Q值就是Q表的Q值
            reward = calculate_reward(quality, rebuffering_ratio)
            
            update(current_state, action, reward, next_state)

        self.qlearning_initial_explore() # 初始化动作已安排，后面应该更新q表

        # 历史数据都在history中，然后再用这些数据来更新Q表，遍历histroy
        history_action = get_history_action()
        history_state = get_history_state()
        history_quality = get_history_quality()
        history_rebuffering_ratio = get_history_rebuffering_ratio()
        

        for i in range(len(history_action)-2):
            train_step(history_state[i], history_action[i], history_state[i+1], history_quality[i], history_rebuffering_ratio[i])


        # init_resolution=1080，init_bitrate_level=2，init_bitrate=71817.751
        nb_segments = len(self.url[self.init_resolution][self.init_bitrate_level]) # 获取清晰度和比特率等级对应的视频片段数量
        latest_selected_arm = 1

        while True:
            i = get_smallest_unscheduled_segment()

            if i > nb_segments:
                # 如果i大于比特率2的视频片段数量，是不是说明可以停止了？
                task1 = dict()
                task1["path_id"] = 1
                task1["eos"] = 1
                task1["initial_explore"] = 0
                task2 = dict()
                task2["path_id"] = 2
                task2["eos"] = 1
                task2["initial_explore"] = 0
                self.download_queue[0].put(task1)
                self.download_queue[1].put(task2)
                break
            
            # DEBUG_SEGMENTS = 200
            if DEBUG and i > DEBUG_SEGMENTS:
                playback_buffer_map[i] = {
                    "eos": 1,
                    "frame_count": 0,
                }
                task1 = dict()
                task1["path_id"] = 1
                task1["eos"] = 1
                task1["initial_explore"] = 0
                task2 = dict()
                task2["path_id"] = 2
                task2["eos"] = 1
                task2["initial_explore"] = 0
                self.download_queue[0].put(task1)
                self.download_queue[1].put(task2)
                break

            if playback_buffer_frame_ratio.value >= 1:
                continue

            if self.scheduled[i] == 1:
                continue
            
            # 这里两个队列对象很牵强，我还以为是一个队列对象对应一个路径，但不是，而是对应一个任务
            # 但这样会让数据更好吗？这里做保留意见，看看日后是否可以优化，如果这样的话只需一个队列对象即可
            # 还不如这样，先决策出一个路径，若另一个路径空闲，则在Q表选择相同状态下的特定路径的最好reward
            # 初始化探索完，开始自动决策加更新，获取当前的state
            path1_statistics = monitor_path(target_ip, network_interface1)
            path2_statistics = monitor_path(target_ip, network_interface2)

            before_request_state = self.discretizer.get_multi_path_state(path1_statistics["rtt"], path1_statistics["throughput"], path2_statistics["rtt"], path2_statistics["throughput"])
            selected_action = choose_action(before_request_state)
            seg_no = i

            path_id, resolution, bitrate, bitrate_level = parse_action(selected_action) # 决策的结果中获取
            path_id = k+1
            task = self.url[resolution][bitrate_level][seg_no] # 这里就已经选取了片段，url信息原来就有，下面则是增加片段信息
            task["seg_no"] = seg_no
            task["resolution"] = resolution
            task["bitrate"] = bitrate
            task["path_id"] = path_id
            task["eos"] = 0
            task["arm"] = selected_action
            task["initial_explore"] = 0
            self.scheduled[seg_no] = 1
            self.download_queue[path_id - 1].put(task)
            
            for k in range(2):
                if self.download_queue[k].qsize() == 0:
                    # 若发现有路径为空，则先预取，预取值为q表对应值,调取的片段为i+5(暂定，这个可以以后调整)
                    unscheduled_path_selected_action = choose_unscheduled_path_action(before_request_state)
                    print("path ", k + 1, " is empty ", " raw action: ", unscheduled_path_selected_action, " seg_no: ", i+5)
                    another_path_id, another_resolution, another_bitrate, another_bitrate_level = parse_action(unscheduled_path_selected_action)

                    task1 = self.url[another_resolution][another_bitrate_level][i+5] # 这里就已经选取了片段，url信息原来就有，下面则是增加片段信息
                    task1["seg_no"] = i + 5
                    task1["resolution"] = another_resolution
                    task1["bitrate"] = another_bitrate
                    task1["path_id"] = another_path_id
                    task1["eos"] = 0
                    task1["arm"] = unscheduled_path_selected_action
                    task1["initial_explore"] = 0
                    self.scheduled[seg_no] = 1
                    self.download_queue[another_path_id - 1].put(task1)


    def contextual_bandit_scheduling(self):
        def get_history_arm():
            # history在download函数中处理
            _history_arm = []
            for x in history:
                path_id = x["path_id"]
                bitrate = x["bitrate"]
                resolution = x["resolution"]
                # 由清晰度和比特率获取比特率水平
                bitrate_level = get_bitrate_level(resolution, bitrate)
                nb_bitrates = get_nb_bitrates()
                _history_arm.append((path_id - 1) * nb_bitrates + bitrate_level)
                # 和task["arm"] = (path_id - 1) * get_nb_bitrates() + k一样的计算方法

            return _history_arm

        def get_history_rtt():
            _history_rtt = [x["rtt"] for x in history]
            return _history_rtt

        def get_latest_rtt():
            return history[len(history) - 1]["rtt"]

        def get_history_throughput():
            _history_throughput = [x["throughput"] for x in history]
            return _history_throughput

        def get_history_mean_throughput_on_path(path_id: int):
            _history_throughput = [x["throughput"] for x in history if x["path_id"] == path_id]
            return numpy.mean(_history_throughput)

        def get_history_mean_rtt_on_path(path_id: int):
            _history_rtt = [x["rtt"] for x in history if x["path_id"] == path_id]
            return numpy.mean(_history_rtt)

        def get_latest_throughput():
            return history[len(history) - 1]["throughput"]

        def get_history_bitrate():
            _history_bitrate = [x["bitrate"] for x in history]
            return _history_bitrate

        def get_latest_bitrate():
            return history[len(history) - 1]["bitrate"]

        def get_latest_bitrate_level():
            bitrate = history[len(history) - 1]["bitrate"]
            resolution = history[len(history) - 1]["resolution"]
            return get_bitrate_level(resolution, bitrate)

        def get_history_rebuffering_ratio():
            _history_rebuffering_ratio = [x["rebuffering_ratio"] for x in history]
            return _history_rebuffering_ratio

        def get_latest_rebuffering_ratio():
            return rebuffering_ratio.value

        def get_history_playback_buffer_ratio():
            _history_playback_buffer_ratio = [x["playback_buffer_ratio"] for x in history]
            return _history_playback_buffer_ratio

        def get_latest_playback_buffer_ratio():
            return playback_buffer_frame_ratio.value
            # return get_playback_buffer_size() * 1.0 / max_playback_buffer_size

        def get_history_reward():
            _history_reward = [x["reward"] for x in history]
            return _history_reward

        def parse_predication(predication) -> (int, int, float, int):
            path_id = (predication - 1) // get_nb_bitrates() + 1
            bitrate_level = (predication - 1) % get_nb_bitrates() + 1

            # get resolution from bitrate_level in bitrate_mapping
            resolution = get_resolution(bitrate_level)
            return path_id, resolution, bitrate_mapping[resolution][bitrate_level], bitrate_level

        def get_smallest_unscheduled_segment():
            for i in range(1, nb_segments):
                if self.scheduled[i] == 0:
                    return i


        # 初始化探索任务，就是布置好探索阶段的任务
        self.initial_explore()

        self.initial_explore_done = True
        """
        history = [
            {
                "arm": <int>,                      # arm编号，表示该任务的选择
                "seg_no": <int>,                   # 视频片段编号，表示当前视频片段的序号
                "resolution": <str>,                # 视频分辨率（如 "1080p", "720p" 等）
                "bitrate": <float>,                # 当前视频的比特率
                "bitrate_ratio": <float>,           # 比特率比例，可能是当前比特率与某个参考比特率的比值
                "throughput": <float>,             # 当前的吞吐量（单位：可能是 Mbps 或 Kbps）
                "rtt": <float>,                    # 往返时间（RTT），这里除以1000并乘以2表示某种转换
                "playback_buffer_ratio": <float>,   # 播放缓冲区比例，可能是播放缓冲区的大小与视频长度的比值
                "rebuffering_ratio": <float>,      # 重缓冲比例，表示播放过程中缓冲的时间占比
                "playback_buffer_ratio_after": <float>,  # 播放缓冲区比例在某个事件后（可能是重缓冲后）
                "rebuffering_ratio_after": <float>, # 重缓冲比例在某个事件后
                "path_id": <int>,                  # 网络路径的 ID，表示数据传输的路径
                "initial_explore": <int>,          # 是否为初步探索（可能用于探索/利用算法中的探索标志）
                "reward": <float>                  # 奖励，可能基于比特率、缓冲情况等因素计算
            },
            # ... more history entries
        ]

        """

        """
        _history_arm = [
            arm_1,  
            arm_2,  
            ...
            arm_n   
        ]
        """
        history_arm = get_history_arm() # 就是获取过去的arm，arm可以表示为决策
        history_reward = get_history_reward()[:len(history_arm)] # 获取过去arm的reward的列表


        train_df = pd.DataFrame({
            "arm": history_arm,
            "bitrate": get_history_bitrate(),
            "throughput": get_history_throughput(),
            "rtt": get_history_rtt(),
            "playback_buffer_ratio": get_history_playback_buffer_ratio()[:len(history_arm)],
            "reward": history_reward,
        })

        scaler = StandardScaler() # 创建sklearn对象
        train = scaler.fit_transform(train_df[["throughput", "rtt", "playback_buffer_ratio"]].values.astype('float64'))
        # 将吞吐量，往返时延，播放缓存率作为一个新的数据类型向量

        # 不同算法，这个alpha可以修改，目前为0.1
        if self.algorithm == "LinUCB":
            self.radius = MAB(
                arms=build_arms(),
                learning_policy=LearningPolicy.LinUCB(alpha=config.linucb_alpha),
            )
            print("Initialized MAB with LinUCB and alpha = ", config.linucb_alpha)
        elif self.algorithm == "LinTS":
            self.radius = MAB(
                arms=build_arms(),
                learning_policy=LearningPolicy.LinTS(alpha=config.lints_alpha),
            )
            print("Initialized MAB with LinTS and alpha = ", config.lints_alpha)
        elif self.algorithm == "LinGreedy":
            self.radius = MAB(
                arms=build_arms(),
                learning_policy=LearningPolicy.LinGreedy(epsilon=config.egreedy_epsilon),
            )
            print("Initialized MAB with LinGreedy and epsilon = ", config.egreedy_epsilon)
        else:
            pass

        # 训练，arm是决策
        self.radius.fit(decisions=train_df["arm"], rewards=train_df["reward"], contexts=train)

        # # init_resolution=1080，init_bitrate_level=2，init_bitrate=71817.751
        nb_segments = len(self.url[self.init_resolution][self.init_bitrate_level]) # 获取清晰度和比特率等级对应的视频片段数量
        latest_selected_arm = 1 
        # (path_id - 1) * get_nb_bitrates() + k  path_id=1,k=1 选择路径1和比特率等级为1的视频片段，即最高码率的视频

        while True:
            i = get_smallest_unscheduled_segment()
            # 下面这些条件判断暂时未知
            if i > nb_segments:
                # 如果i大于比特率2的视频片段数量，是不是说明可以停止了？
                task1 = dict()
                task1["path_id"] = 1
                task1["eos"] = 1
                task1["initial_explore"] = 0
                task2 = dict()
                task2["path_id"] = 2
                task2["eos"] = 1
                task2["initial_explore"] = 0
                self.download_queue[0].put(task1)
                self.download_queue[1].put(task2)

                break
            
            # DEBUG_SEGMENTS = 200
            if DEBUG and i > DEBUG_SEGMENTS:
                playback_buffer_map[i] = {
                    "eos": 1,
                    "frame_count": 0,
                }
                task1 = dict()
                task1["path_id"] = 1
                task1["eos"] = 1
                task1["initial_explore"] = 0
                task2 = dict()
                task2["path_id"] = 2
                task2["eos"] = 1
                task2["initial_explore"] = 0
                self.download_queue[0].put(task1)
                self.download_queue[1].put(task2)
                break

            if playback_buffer_frame_ratio.value >= 1:
                continue

            if self.scheduled[i] == 1:
                continue

            for k in range(2):
                if self.download_queue[k].qsize() == 0:
                    # 队列对象长度为0，即路径空闲，这时进行路径预测并添加任务调度
                    test_df = pd.DataFrame({
                        "throughput": [get_history_mean_throughput_on_path(k + 1)], # 获取每个路径的平均吞吐量
                        "rtt": [get_history_mean_rtt_on_path(k + 1)], # 获取每个路径的平均rtt
                        "playback_buffer_ratio": [get_latest_playback_buffer_ratio()], # 这里得替换
                    })
                    
                    # 数据预处理
                    test = scaler.transform(test_df.values.astype('float64'))
                    if k == 0: 
                        prediction = self.radius.predict(test)
                        # partial_fit is done after download finished, because only then we know the reward
                        latest_selected_arm = prediction
                        seg_no = i
                        print("path ", k + 1, " is empty ", " raw prediction: ", prediction, " seg_no: ", i)
                    else:
                        prediction = latest_selected_arm 
                        # 不是直接安排最大的码率传输，而是复用，先预取i+5的片段调度，码率和路径和之前一样
                        seg_no = i + 5
                        print("path ", k + 1, " is empty ", " raw prediction: ", prediction, " seg_no: ", i+5)
                    
                    # 这里的两个队列对象只是判断路径是否有任务，具体调度则是下面代码决定
                    path_id, resolution, bitrate, bitrate_level = parse_predication(prediction) # 决策的结果中获取
                    raw_prediction = prediction
                    path_id = k+1
                    task = self.url[resolution][bitrate_level][seg_no] # 这里就已经选取了片段，url信息原来就有，下面则是增加片段信息
                    task["seg_no"] = seg_no
                    task["resolution"] = resolution
                    task["bitrate"] = bitrate
                    task["path_id"] = path_id
                    task["eos"] = 0
                    task["arm"] = prediction
                    task["context"] = test
                    task["initial_explore"] = 0
                    self.scheduled[seg_no] = 1
                    self.download_queue[path_id - 1].put(task)
                    time.sleep(0.1)
                    print(
                        f"{bcolors.RED}{bcolors.BOLD}seg_no: %d, {bcolors.WARNING}raw predication: %d, selected arm: "
                        f"%d, path_id: %d,"
                        f"resolution: %d, bitrate_level: %d{bcolors.ENDC}" % (
                            seg_no, raw_prediction, prediction, path_id, resolution,
                            bitrate_level))


# if __name__ == "__main__":
    # path1_result = monitor_path(target_ip, network_interface1)
    # path2_result = monitor_path(target_ip, network_interface2)

    # print(f"RTT (ms): {path1_result['rtt']}")
    # print(f"吞吐量 (KB/s): {path1_result['throughput']}")

    # print(f"RTT (ms): {path2_result['rtt']}")
    # print(f"吞吐量 (KB/s): {path2_result['throughput']}")



