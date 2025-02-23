import argparse
from multiprocessing import Process

from config import init
from downloader import Downloader
from player import MockPlayer

# 目前已知，用picoquic搭建quic服务器（和客户端，存疑），再加上运用mpd文件，来实现基于HTTP3的DASH协议，来传输流媒体。

if __name__ == "__main__":
    global exp_id, algorithm, linucb_alpha, lints_alpha, egreedy_epsilon, DEBUG_SEGMENTS

    parser = argparse.ArgumentParser(description="Adaptive video streaming with contextual bandits and MPQUIC")
    parser.add_argument("--exp_id", default="default", help="experiment id")
    parser.add_argument("--scheduler", choices=["roundrobin", "minrtt", "contextual_bandit"], default="contextual_bandit", help="scheduler")
    parser.add_argument("--algorithm", choices=["LinUCB", "LinTS", "LinGreedy"], default="LinUCB", help="scheduling algorithm")
    parser.add_argument("--linucb_alpha", type=float, default=0.1, help="alpha for LinUCB")
    parser.add_argument("--lints_alpha", type=float, default=0.1, help="alpha for LinTS")
    parser.add_argument("--egreedy_epsilon", type=float, default=0.1, help="epsilon for epsilon-greedy")
    parser.add_argument("--nb_segment", type=int, default=100, help="number of segments to download")

    args = parser.parse_args()

    # 参数变为全局变量，确保整个程序都能访问，用于进程间共享
    init(args.exp_id, args.algorithm, args.linucb_alpha, args.lints_alpha, args.egreedy_epsilon, args.nb_segment)

    # 开启进程
    mplayer = MockPlayer(exp_id=args.exp_id)
    mplayer_process = Process(target=mplayer.play)
    mplayer_process.start()


    mdownloader = Downloader(scheduler=args.scheduler, algorithm=args.algorithm)
    mdownload_process = Process(target=mdownloader.main_loop)
    mdownload_process.start()

    mplayer_process.join()
    mdownload_process.join()
