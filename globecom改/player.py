import json
import time

from config import playback_buffer_map, rebuffering_ratio, history, playback_buffer_frame_count, \
    playback_buffer_frame_ratio


class MockPlayer:
    def __init__(self, exp_id: str):
        self.played_segments = 0 # 播放的视频片段
        self.downloaded_segments = 0 # 下载的视频片段
        self.ts_playback_start = time.time() # 播放开始时间
        self.rebuffering_event = list() # 存储每一次的重缓存事件，包括开始和结束时间，下一个片段，是否完成
        self.experiment_id = exp_id

    def get_total_rebuffered_time(self):
        total = 0
        for x in self.rebuffering_event:
            if x["completed"] == 1:
                # 如果重缓存完成，则end - start
                total += (x["end"] - x["start"])
            else:
                # 否则当前时间-start
                total += (time.time() - x["start"])
        return total

    # print statistics
    def print_statistics(self):
        print(history)
        t = str(int(time.time()))
        # write history to file with timestamp in filename
        if len(self.experiment_id) > 0:
            filename = "result/history_" + self.experiment_id + "_" + t + ".json"
        else:
            filename = "result/history_" + t + ".json"
        with open(filename, "w") as f:
            f.write(json.dumps(history._getvalue(), indent=4))
        print("history written to %s" % filename)

        print(self.rebuffering_event)
        if len(self.experiment_id) > 0:
            filename = "result/rebuffering_history_" + self.experiment_id + "_" + t + ".json"
        else:
            filename = "result/rebuffering_history_" + t + ".json"
        with open(filename, "w") as f:
            f.write(json.dumps(self.rebuffering_event, indent=4))

        print("history rebuffering event written to %s" % filename)

    def count_current_playable_frames(self, current_seg_no):
        playable_frames = 0
        keys = playback_buffer_map.keys()
        for i in keys:
            playable = True
            # 从当前帧的下一帧到第i帧，看是否有帧缺失
            for j in range(current_seg_no + 1, i + 1):
                if j not in keys:
                    playable = False
                    break
            if playable:
                playable_frames += playback_buffer_map[i]["frame_count"]
        return playable_frames

    def play(self):
        while True:
            # 重缓存比例的值，分母是当前时间-开始时间，分子是总的重缓存时间
            rebuffering_ratio.value = self.get_total_rebuffered_time() * 1.0 / (time.time() - self.ts_playback_start)
            # 能播放缓存帧数量
            playback_buffer_frame_count.value = self.count_current_playable_frames(self.played_segments + 1)
            # 播放缓存帧比率，共240帧
            playback_buffer_frame_ratio.value = playback_buffer_frame_count.value / 240

            if self.played_segments + 1 in playback_buffer_map.keys():
                # if the next segment is in the playback buffer, play it
                # 感觉这个是当下一个片段可以播放时，就停止重缓存？
                # 这里的重缓存是指下一个视频段没有缓存完全的时间？
                # 系统通过检查片段是否可以播放来跟踪重缓存事件。如果一个片段还没有准备好（即不在 playback_buffer_map 中），就会触发一个重缓存事件，记录开始时间。
                # 当该片段缓存完成并准备好播放时，记录重缓存事件的结束时间。
                if len(self.rebuffering_event) > 0 and self.rebuffering_event[-1]["completed"] == 0:
                    # 下一帧可播放且重缓存的最后一次事件标记为0，则记录终止时间，并标记改为1
                    self.rebuffering_event[-1]["end"] = time.time()
                    self.rebuffering_event[-1]["completed"] = 1

                task = playback_buffer_map[self.played_segments + 1]
                # if this is the last segment, stop the playback
                if task["eos"] == 1:
                    print("EOS")
                    break

                frames = task["frame_count"]

                print("playing segment ", self.played_segments + 1,
                      " with ", frames, " frames",
                      " current buffer size: ", len(playback_buffer_map),
                      sorted(playback_buffer_map.keys()))

                # 模拟播放视频帧
                for i in range(int(frames)):
                    time.sleep(1.0 / 24) # 模拟24帧每秒（FPS）的播放速度
                    # 记录当前剩余的可播放帧数。更新了播放缓冲区的状态。
                    playback_buffer_frame_count.value = (task["frame_count"] - i - 1) + self.count_current_playable_frames(self.played_segments + 1) * 1.0
                    playback_buffer_frame_ratio.value = playback_buffer_frame_count.value / 240

                self.played_segments += 1
                del playback_buffer_map[self.played_segments] # 删除已播放的帧
            else:
                if len(self.rebuffering_event) == 0 or self.rebuffering_event[-1]["completed"] == 1:
                    # 就是未发生重缓存或者上一个重缓存完成，则开始计时，当下一个视频段缓存完成时，计时结束。
                    self.rebuffering_event.append({
                        "start": time.time(),
                        "end": -1,
                        "next_seg_no": self.played_segments + 1,
                        "completed": 0
                    })
                    print("######## New rebuffering event added at %f for seg %d" %
                          (self.rebuffering_event[-1]["start"],
                           self.rebuffering_event[-1]["next_seg_no"]))
                else:
                    # calculate rebuffering ratio
                    rebuffering_ratio.value = self.get_total_rebuffered_time() * 1.0 / (
                                time.time() - self.ts_playback_start)

        # end of playback
        self.rebuffering_event.append({
            "start": time.time(),
            "end": time.time(),
            "next_seg_no": -1,
            "completed": 1
        }) # 给列表加上字典，包含开始时间，结束时间，下一个片段编号，是否完成
        self.print_statistics()  # 输出数据
