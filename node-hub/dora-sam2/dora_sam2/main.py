import cv2
import numpy as np
import pyarrow as pa
import torch
from dora import Node
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 手动下载方法（保持原有代码不变）：
# 1. 访问模型仓库 https://huggingface.co/facebook/sam2-hiera-large
# 2. 下载整个仓库（推荐使用git）
#    git lfs install
#    git clone https://huggingface.co/facebook/sam2-hiera-large
# 3. 将模型文件放入缓存目录（默认路径）：
#    Windows: C:\Users\你的用户名\.cache\huggingface\hub\models--facebook--sam2-hiera-large
#    Linux/Mac: ~/.cache/huggingface/hub/models--facebook--sam2-hiera-large

# 或指定自定义缓存路径（添加环境变量）：
import os
# os.environ['TRANSFORMERS_CACHE'] = '/media/cheku/c5cb6806-c194-47de-994b-583ebeca393f2/wjp/sam2'  # 添加在代码最前面
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")

def main():
    pa.array([])  # initialize pyarrow array
    node = Node()
    frames = {}
    last_pred = None
    labels = None
    return_type = pa.Array
    image_id = None
    for event in node:
        event_type = event["type"]

        if event_type == "INPUT":
            event_id = event["id"]

            if "image" in event_id:
                storage = event["value"]
                metadata = event["metadata"]
                encoding = metadata["encoding"]
                width = metadata["width"]
                height = metadata["height"]

                if (
                    encoding == "bgr8"
                    or encoding == "rgb8"
                    or encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]
                ):
                    channels = 3
                    storage_type = np.uint8
                else:
                    error = f"Unsupported image encoding: {encoding}"
                    raise RuntimeError(error)

                if encoding == "bgr8":
                    frame = (
                        storage.to_numpy()
                        .astype(storage_type)
                        .reshape((height, width, channels))
                    )
                    frame = frame[:, :, ::-1]  # OpenCV image (BGR to RGB)
                elif encoding == "rgb8":
                    frame = (
                        storage.to_numpy()
                        .astype(storage_type)
                        .reshape((height, width, channels))
                    )
                elif encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:
                    storage = storage.to_numpy()
                    frame = cv2.imdecode(storage, cv2.IMREAD_COLOR)
                    frame = frame[:, :, ::-1]  # OpenCV image (BGR to RGB)
                else:
                    raise RuntimeError(f"Unsupported image encoding: {encoding}")
                image = Image.fromarray(frame)
                frames[event_id] = image

                # TODO: Fix the tracking code for SAM2.
                continue
                if last_pred is not None:
                    with (
                        torch.inference_mode(),
                        torch.autocast(
                            "cuda",
                            dtype=torch.bfloat16,
                        ),
                    ):
                        predictor.set_image(frames[image_id])

                        new_logits = []
                        new_masks = []

                        if len(last_pred.shape) < 3:
                            last_pred = np.expand_dims(last_pred, 0)

                        for mask in last_pred:
                            mask = np.expand_dims(mask, 0)  # Make shape: 1x256x256
                            masks, _, new_logit = predictor.predict(
                                mask_input=mask,
                                multimask_output=False,
                            )
                            if len(masks.shape) == 4:
                                masks = masks[:, 0, :, :]
                            else:
                                masks = masks[0, :, :]

                            masks = masks > 0
                            new_masks.append(masks)
                            new_logits.append(new_logit)
                            ## Mask to 3 channel image

                        last_pred = np.concatenate(new_logits, axis=0)
                        masks = np.concatenate(new_masks, axis=0)

                        match return_type:
                            case pa.Array:
                                node.send_output(
                                    "masks",
                                    pa.array(masks.ravel()),
                                    metadata={
                                        "image_id": image_id,
                                        "width": frames[image_id].width,
                                        "height": frames[image_id].height,
                                    },
                                )
                            case pa.StructArray:
                                node.send_output(
                                    "masks",
                                    pa.array(
                                        [
                                            {
                                                "masks": masks.ravel(),
                                                "labels": event["value"]["labels"],
                                            },
                                        ],
                                    ),
                                    metadata={
                                        "image_id": image_id,
                                        "width": frames[image_id].width,
                                        "height": frames[image_id].height,
                                    },
                                )

            if "boxes2d" in event_id:

                if isinstance(event["value"], pa.StructArray):
                    # 添加copy()创建可写副本
                    boxes2d = event["value"][0].get("bbox").values.to_numpy(zero_copy_only=False).copy()
                    labels = (
                        event["value"][0]
                        .get("labels")
                        .values.to_numpy(zero_copy_only=False)
                    )
                    return_type = pa.Array
                else:
                    # 添加copy()创建可写副本
                    boxes2d = event["value"].to_numpy().copy()
                    labels = None
                    return_type = pa.Array

                metadata = event["metadata"]
                encoding = metadata["encoding"]
                if encoding != "xyxy":
                    raise RuntimeError(f"Unsupported boxes2d encoding: {encoding}")
                boxes2d = boxes2d.reshape(-1, 4)  # 将坐标转换为(x1, y1, x2, y2)格式
                # rgb_image_width=1280
                # rgb_image_height=720
                
                # width = metadata.get("width", 848)    # 添加默认值
                # height = metadata.get("height", 480)
                
                # scale_width=width/rgb_image_width
                # scale_height=height/rgb_image_height
                
                # # 修改为逐元素处理
                # boxes2d = boxes2d * np.array([scale_width, scale_height, scale_width, scale_height])
                
                image_id = metadata["image_id"]
                with (
                    torch.inference_mode(),
                    torch.autocast(
                        "cuda",
                        dtype=torch.bfloat16,
                    ),
                ):
                    predictor.set_image(frames[image_id])
                    masks, _scores, last_pred = predictor.predict(
                        box=boxes2d, point_labels=labels, multimask_output=False,
                    )

                    if len(masks.shape) == 4:
                        masks = masks[:, 0, :, :]
                        last_pred = last_pred[:, 0, :, :]
                    else:
                        masks = masks[0, :, :]
                        last_pred = last_pred[0, :, :]

                    masks = masks > 0
                    ## Mask to 3 channel image
                    match return_type:
                        case pa.Array:
                            node.send_output(
                                "masks",
                                pa.array(masks.ravel()),
                                metadata={
                                    "image_id": image_id,
                                    "width": frames[image_id].width,
                                    "height": frames[image_id].height,
                                },
                            )
                        case pa.StructArray:
                            node.send_output(
                                "masks",
                                pa.array(
                                    [
                                        {
                                            "masks": masks.ravel(),
                                            "labels": event["value"]["labels"],
                                        },
                                    ],
                                ),
                                metadata={
                                    "image_id": image_id,
                                    "width": frames[image_id].width,
                                    "height": frames[image_id].height,
                                },
                            )

        elif event_type == "ERROR":
            print("Event Error:" + event["error"])


if __name__ == "__main__":
    main()