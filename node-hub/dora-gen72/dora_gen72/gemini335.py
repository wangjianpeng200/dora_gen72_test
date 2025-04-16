import os
import sys
import numpy as np
import pyarrow as pa
from dora import Node
from pyorbbecsdk import Pipeline, Config, OBSensorType, FrameSet,OBFormat,OBError

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils import frame_to_bgr_image
import atexit

def main():
    node = Node()
    pipeline = Pipeline()
    config = Config()

    try:
        # 配置深度传感器（修改部分）
        depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        try:
            # 指定1280x720分辨率，Y16格式，30fps
            depth_profile = depth_profile_list.get_video_stream_profile(1280, 720, OBFormat.Y16, 30)
        except OBError as e:
            print(f"不支持1280x720配置，使用默认配置: {e}")
            depth_profile = depth_profile_list.get_default_video_stream_profile()
        
        config.enable_stream(depth_profile)
        depth_width = depth_profile.get_width()
        depth_height = depth_profile.get_height()

        # 配置彩色传感器
        color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)
        color_width = color_profile.get_width()
        color_height = color_profile.get_height()

        pipeline.start(config)
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    finally:
        atexit.register(pipeline.stop)

    # 获取相机内参
    camera_param = pipeline.get_camera_param()
    K = np.array([
        [camera_param.depth_intrinsic.fx, 0, camera_param.depth_intrinsic.cx],
        [0, camera_param.depth_intrinsic.fy, camera_param.depth_intrinsic.cy],
        [0, 0, 1]
    ])

    for event in node:
        if event["type"] == "INPUT" and event["id"] == "tick":
            try:
                frames = pipeline.wait_for_frames(100)
            except Exception as e:
                print(f"获取帧失败: {e}")
                continue

            # 检查 frames 是否为 None
            if frames is None:
                print("未获取到有效帧，跳过本次处理")
                continue

            # 处理彩色帧
            color_frame = frames.get_color_frame()
            if color_frame is None:
                print("未获取到彩色帧")
                continue
            color_image = frame_to_bgr_image(color_frame)
            
            # 新增RGB尺寸打印
            # print(f"RGB图像尺寸: {color_width}x{color_height}")
            
            # 发送彩色图像
            node.send_output(
                "image_depth",
                pa.array(color_image.ravel()),
                metadata={
                    "encoding": "bgr8",
                    "width": color_width,
                    "height": color_height,
                },
            )

            # 处理深度帧
            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                print("未获取到深度帧")
                continue
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((depth_height, depth_width))
            
            # 新增深度尺寸打印
            # print(f"深度图像尺寸: {depth_width}x{depth_height}")
            
            scale = depth_frame.get_depth_scale()
            # print(f"深度比例: {scale}")
            # [修改点] 转换深度数据为 float64 并优化内存布局
            depth_float = (depth_data.astype(np.float64) * 0.001)  # 改为双精度
            # print(f"深度数据类型: {depth_float.dtype}")
            depth_float = np.ascontiguousarray(depth_float)  # 确保内存连续

            # 发送深度数据
            node.send_output(
                "depth",
                pa.array(depth_float.ravel()),
                metadata={
                    "width": depth_width,
                    "height": depth_height,
                    "focal": [int(K[0, 0]), int(K[1, 1])],
                    "resolution": [int(K[0, 2]), int(K[1, 2])],
                },
            )

if __name__ == "__main__":
    main()