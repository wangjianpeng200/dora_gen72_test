"""TODO: Add docstring."""

# State Machine
import json
import os
import time

import numpy as np
import pyarrow as pa
from dora import Node

IMAGE_RESIZE_RATIO = float(os.getenv("IMAGE_RESIZE_RATIO", "1.0"))
node = Node()

ACTIVATION_WORDS = os.getenv("ACTIVATION_WORDS", "").split()
TABLE_HEIGHT = float(os.getenv("TABLE_HEIGHT", "-0.41"))

l_init_pose = [
    -19.323,
    36.125,
    -4.361,
    -101.783, 
    74.795,
    -12.8,
    -20.141,
    100,
] 

# l_init_pose = [
#     -19.323,
#     31.125,
#     -8.361,
#     -116.783, 
#     97.795,
#     3.08,
#     -15.141,
#     100,
# ] 



r_init_pose = [
    24.15,
    41.65,
    -0.757,
    -95.508,
    -78.867,
    -18.52,
    49,
    100,
]

# right_R=np.array([[ 0.03041012, -0.75495311,  0.6550733 ],
#                [ 0.01426148, -0.65498197, -0.75550991],
#                [ 0.99943576,  0.03231746, -0.00915133]])

# right_t= np.array([[ 0.10971697],
#                [ 0.16180161],
#                [-0.06287235]])

right_R=np.array([[ 0.0663028,  -0.80993549,  0.58275933],
               [ 0.00654148, -0.5836791,  -0.81195807],
               [ 0.99777811,  0.0576472,  -0.03340138]])

right_t= np.array([[ 0.11050572],
               [ 0.15650712],
               [-0.05664732]])


left_R=np.array([[ 0.01247585, -0.82119797,  0.57050701],
               [-0.07766833, 0.5680318,   0.81933357],
               [-0.99690119, -0.05453221, -0.0566944 ]])

left_t= np.array([[ 0.13283284],
               [-0.165495],
               [-0.05372739]])

stop = True




def camera_to_base(point_camera,R,t):
    point_camera = np.array(point_camera).reshape(3,1)
    point_base = np.dot(R, point_camera) + t
    return point_base.flatten() 

def extract_bboxes(json_text) -> (np.ndarray, np.ndarray):
    """Extract bounding boxes from a JSON string with markdown markers and return them as a NumPy array.

    Parameters
    ----------
    json_text : str
        JSON string containing bounding box data, including ```json markers.

    Returns
    -------
    np.ndarray: NumPy array of bounding boxes.

    """
    # Ensure all lines are stripped of whitespace and markers
    lines = json_text.strip().splitlines()

    # Filter out lines that are markdown markers
    clean_lines = [line for line in lines if not line.strip().startswith("```")]

    # Join the lines back into a single string
    clean_text = "\n".join(clean_lines)
    # Parse the cleaned JSON text
    try:
        data = json.loads(clean_text)

        # Extract bounding boxes
        bboxes = [item["bbox_2d"] for item in data]
        labels = [item["label"] for item in data]

        return np.array(bboxes), np.array(labels)
    except Exception as _e:  # noqa
        pass
    return None, None


def handle_speech(last_text):
    """TODO: Add docstring."""
    global stop
    word_list=list(last_text.lower())
    words = last_text.lower().split()
    print("words: ", words)
    if len(ACTIVATION_WORDS) > 0 and any(word in ACTIVATION_WORDS for word in words):

        node.send_output(
            "text_vlm",
            pa.array(
                [
                    f"Given the prompt: {cache['text']}. Output the two bounding boxes for the two objects",
                ],
            ),
            metadata={"image_id": "image_depth"},
        )
        node.send_output(
            "prompt",
            pa.array([cache["text"]]),
            metadata={"image_id": "image_depth"},
        )
        print(f"sending: {cache['text']}")
        stop = False


def wait_for_event(id, timeout=None, cache={}):
    """TODO: Add docstring."""
    while True:
        event = node.next(timeout=timeout)
        if event is None:
            cache["finished"] = True
            return None, cache
        if event["type"] == "INPUT":
            cache[event["id"]] = event["value"]
            if event["id"] == "text":
                cache[event["id"]] = event["value"][0].as_py()
                handle_speech(event["value"][0].as_py())
            elif event["id"] == id:
                return event["value"], cache

        elif event["type"] == "ERROR":
            return None, cache


def wait_for_events(ids: list[str], timeout=None, cache={}):
    """TODO: Add docstring."""
    response = {}
    while True:
        event = node.next(timeout=timeout)
        if event is None:
            cache["finished"] = True
            return None, cache
        if event["type"] == "INPUT":
            cache[event["id"]] = event["value"]
            if event["id"] == "text":
                cache[event["id"]] = event["value"][0].as_py()
                handle_speech(event["value"][0].as_py())
            elif event["id"] in ids:
                response[event["id"]] = event["value"]
                if len(response) == len(ids):
                    return response, cache
        elif event["type"] == "ERROR":
            return None, cache


def get_prompt():
    """TODO: Add docstring."""
    text = wait_for_event(id="text", timeout=0.3)
    if text is None:
        return None
    text = text[0].as_py()
    words = text.lower().split()
    print("words: ", words)
    # Check if any activation word is present in the text
    if len(ACTIVATION_WORDS) > 0 and all(
        word not in ACTIVATION_WORDS for word in words
    ):
        return None
    return text


last_text = ""
cache = {"text": "Put the orange in the metal box"}

while True:
    ### === IDLE ===

    node.send_output(
        "action_r_arm",
        pa.array(r_init_pose),
        metadata={"encoding": "jointstate", "duration": 1,"arm":"right"},
    )
    time.sleep(0.5)
    node.send_output(
        "action_l_arm",
        pa.array(l_init_pose),
        metadata={"encoding": "jointstate", "duration": 1,"arm":"left"},
    )
    _, cache = wait_for_events(
        ids=["response_r_arm", "response_l_arm"], timeout=3, cache=cache,
    )
    # handle_speech(cache["text"])

    ### === TURNING ===

    # Trigger action once text from whisper is received
    # Move left. Overwrite this with your desired movement..
    # node.send_output("action_base", pa.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.57]))
    # Look straight
    # node.send_output("look", pa.array([0.3, 0, -0.1]))
    # You can add additional actions here
    # ...

    # event = wait_for_event(id="response_base")[0].as_py()
    # if not event:
    ## return to IDLE
    # node.send_output("action_base", pa.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.57]))
    # event = wait_for_event(id="response_base")[0].as_py()
    # if event:
    # continue
    # else:
    # break

    ### === GRABBING ===

    # Trigger action once base is done moving
    # node.send_output(
    # "text_vlm",
    # pa.array([f"Given the prompt: {text}. Output bounding box for this action"]),
    # metadata={"image_id": "image_depth"},
    # )
    arm_holding_object = None
    # Try pose and until one is successful
    text, cache = wait_for_event(id="text", timeout=0.3, cache=cache)

    if stop:
        continue

    while True:
        values, cache = wait_for_event(id="pose", cache=cache)

        if values is None:
            continue
        values = values.to_numpy().reshape((-1, 6))
        if len(values) < 2:
            continue
        x = values[0][0]
        y = values[0][1]
        z = values[0][2]
        dest_x = values[1][0]
        dest_y = values[1][1]
        dest_z = values[1][2]

        ## Clip the Maximum and minim values for the height of the arm to avoid collision or weird movement.

        if x > 0:
            x_new,y_new,z_new = camera_to_base([x,y,z],right_R,right_t) 
            trajectory = np.array([
                [x_new, -0.07, z_new, 0, 0, 0, 100],
                [x_new, y_new+0.003, z_new, 0, 0, 0, 0],
                [x_new, -0.07, z_new, 0, 0, 0, 0],
            ]).ravel() 
            node.send_output(
                "action_r_arm",
                pa.array(trajectory),
                metadata={"encoding": "xyzrpy", "duration": "0.5","arm":"right"},
            )
            event, cache = wait_for_event(id="response_r_arm", timeout=12, cache=cache)
            if event is not None and event[0].as_py():
                print("Success")
                arm_holding_object = "right"
                break
            else:
                print("Failed: x: ", x_new, " y: ", y_new, " z: ", z_new)
                node.send_output(
                    "action_r_arm",
                    pa.array(r_init_pose),
                    metadata={"encoding": "jointstate", "duration": "0.75","arm":"right"},
                )
                event, cache = wait_for_event(id="response_r_arm", cache=cache)
        else:
            x_new,y_new,z_new = camera_to_base([x,y,z],left_R,left_t)
            trajectory = np.array([
                [x_new, 0.05, z_new, 0, 0, 0, 100],
                [x_new, y_new+0.008, z_new, 0, 0, 0, 0],
                [x_new, 0.05, z_new, 0, 0, 0, 0],
            ]).ravel()
            node.send_output(
                "action_l_arm",
                pa.array(trajectory),
                metadata={"encoding": "xyzrpy", "duration": "0.5","arm":"left"},
            )
            event, cache = wait_for_event(id="response_l_arm", timeout=12, cache=cache)
            if event is not None and event[0].as_py():
                print("Success")
                arm_holding_object = "left"
                break
            else:
                print("Failed: x: ", x_new, " y: ", y_new, " z: ", z_new)
                node.send_output(
                    "action_l_arm",
                    pa.array(l_init_pose),
                    metadata={"encoding": "jointstate", "duration": "0.75","arm":"left"},
                )
                event, cache = wait_for_event(id="response_l_arm", cache=cache)
    ### === RELEASING ===

    # Trigger action once r_arm is done moving
    # node.send_output("action_base", pa.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.57]))
    # event = wait_for_event(id="response_base")[0].as_py()

    # if not event:
    #    print("Failed to move right")

    # Trigger action to release object
    if arm_holding_object == "right":
        dest_x_new,dest_y_new,dest_z_new = camera_to_base([dest_x,dest_y,dest_z],right_R,right_t)
        node.send_output(
            "action_r_arm",
            pa.array(
                [
                    dest_x_new,
                    -0.008,
                    dest_z_new,
                    0,
                    0,
                    0,
                    100,
                ],
            ),
           metadata={"encoding": "xyzrpy", "duration": "0.75","arm":"right"},
        )
        event, cache = wait_for_event(id="response_r_arm", cache=cache)
    else:
        dest_x_new,dest_y_new,dest_z_new = camera_to_base([dest_x,dest_y,dest_z],left_R,left_t)
        node.send_output(
            "action_l_arm",
            pa.array(
                [
                    dest_x_new,
                    0.16,
                    dest_z_new,
                    0,
                    0,
                    0,
                    100,
                ],
            ),
            metadata={"encoding": "xyzrpy", "duration": "0.75","arm":"left"},
        )
        event, cache = wait_for_event(id="response_l_arm", cache=cache)

    if event is None or not event[0].as_py():
        print("Failed to release object")
        if arm_holding_object == "right":
            node.send_output(
                "action_r_arm",
                pa.array(
                    [
                        x_new,
                        y_new,
                        z_new,
                        0,
                        0,
                        0,
                        100,
                    ],
                ),
               metadata={"encoding": "xyzrpy", "duration": "0.75","arm":"right"},
            )
            event, cache = wait_for_event(id="response_r_arm", cache=cache)
        else:
            node.send_output(
                "action_l_arm",
                pa.array(
                    [
                        x_new,
                        y_new,
                        z_new,
                        0,
                        0,
                        0,
                        100,
                    ],
                ),
                metadata={"encoding": "xyzrpy", "duration": "0.75","arm":"left"},
            )
            event, cache = wait_for_event(id="response_l_arm", cache=cache)
    else:
        stop = True

    if cache.get("finished", False):
        break
    # Move object back to initial position
