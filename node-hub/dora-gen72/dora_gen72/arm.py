import os
import time
import sys
import numpy as np
import pyarrow as pa
from dora import Node

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from Robotic_Arm.rm_robot_interface import *

RIGHT_ROBOT_IP = os.getenv("ROBOT_IP", "192.168.1.18")
LEFT_ROBOT_IP = os.getenv("ROBOT_IP", "192.168.1.19")

l_init_pose = [
    -105.478,
    81.933,
    85.075,
    -100.802,
    -10.81,
    0.654,
    3.414,
    100,
]
r_init_pose = [
    12.15,
    36.437,
    14.391,
    -95.508,
    -91.726,
    -23.781,
    0,
    100,
]

class robot_gen72:
    def __init__(self):
        #---------------------------------connect
        self.right_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.left_arm= RoboticArm()
        right_connect=self.right_arm.rm_create_robot_arm("192.168.1.19", 8080)
        left_connect= self.left_arm.rm_create_robot_arm("192.168.1.18", 8080)
        #---------------------------------right_arm init
        self.right_arm.rm_set_arm_run_mode(2)
        self.right_arm.rm_set_tool_voltage(3) 
        self.right_arm.rm_set_modbus_mode(1,115200,2)   
        self.right_write_params = rm_peripheral_read_write_params_t(1, 40000, 1)
        self.right_arm.rm_write_single_register(self.right_write_params,100) 
        #---------------------------------left_arm init
        self.left_arm.rm_set_tool_voltage(3) 
        self.left_arm.rm_set_modbus_mode(1,115200,2)
        self.left_write_params = rm_peripheral_read_write_params_t(1, 40000, 1)
        self.left_arm.rm_write_single_register(self.left_write_params,100)
        self.left_arm.rm_change_tool_frame("lebai2")

def main():
    node = Node()  
    gen72=robot_gen72()
    for event in node:  
        if event["type"] == "INPUT" and (event["id"] == "pose_r" or event["id"]=="pose_l"):  # 修正缩进：统一使用4个空格
            values: np.array = event["value"].to_numpy(zero_copy_only=False)
            encoding = event["metadata"]["encoding"]
            wait = event["metadata"].get("wait", True)       
            duration = float(event["metadata"].get("duration", 1))  
            grasp = event["metadata"].get("grasp", True)    
            arm_p = event["metadata"].get("arm","right")
            if encoding == "xyzrpy":
                values = values.reshape((-1, 7))
                for i, value in enumerate(values):
                    print(f"Processing row {i+1}/{len(values)}")
                    pose = [
                        float(value[0]),  
                        float(value[1]), 
                        float(value[2]), 
                        float(value[3]),  
                        float(value[4]), 
                        float(value[5])   
                    ]
                    gripper = float(value[6])
                    if arm_p=="right":
                        error_code, curr_state = gen72.right_arm.rm_get_current_arm_state()  
                        curr_pose = curr_state['pose']        
                        pose[3]=float(curr_pose[3])
                        pose[4]=float(curr_pose[4])
                        pose[5]=float(curr_pose[5])
                        move_p=gen72.right_arm.rm_movej_p(pose, 20, 0, 0, 1)
                        print("第", i+1, "次right arm move_p:", pose)
                        time.sleep(0.5)
                        gen72.right_arm.rm_write_single_register(
                            gen72.right_write_params,
                            int(gripper))
                        time.sleep(1.0)
                        if move_p != 0:
                            node.send_output(  
                                "response_r_arm",
                                pa.array([False]),
                                metadata={"error": "Failed to grasp"},
                            )
                            break  
                    else:
                        error_code, curr_state = gen72.left_arm.rm_get_current_arm_state()
                        curr_pose = curr_state['pose']
                        pose[3]=float(curr_pose[3])
                        pose[4]=float(curr_pose[4])
                        pose[5]=float(curr_pose[5])
                        move_p=gen72.left_arm.rm_movej_p(pose, 20, 0, 0, 1)
                        print("第", i+1, "次left arm move_p:", pose)
                        time.sleep(0.5)
                        print(gripper)
                        gen72.left_arm.rm_write_single_register(
                            gen72.left_write_params,
                            int(gripper))
                        time.sleep(1.0)
                        if move_p != 0:
                            node.send_output(
                                "response_l_arm",
                                pa.array([False]),
                                metadata={"error": "Failed to grasp"},
                            )
                            break
                if move_p == 0:
                    if arm_p=="right":
                        node.send_output("response_r_arm", pa.array([True]))  
                    else:
                        node.send_output("response_l_arm", pa.array([True]))  
      
            elif encoding == "jointstate":
                values = values.reshape((-1, 8))
                arm_j=event["metadata"].get("arm","right")
                for value in values:
                    joints = value[:7].tolist()  
                    gripper = value[7]          
                    if arm_p=="right":
                        move_j=gen72.right_arm.rm_movej(joints, 20, 0, 0, 1)           
                        response_gripper = gen72.right_arm.rm_write_single_register(gen72.right_write_params,int(gripper))
                    else:
                        move_j=gen72.left_arm.rm_movej(joints, 20, 0, 0, 1)
                        response_gripper = gen72.left_arm.rm_write_single_register(gen72.left_write_params,int(gripper))
                if move_j==0:
                    if arm_p=="right":
                        node.send_output("response_r_arm", pa.array([True]))  
                    else:
                        node.send_output("response_l_arm", pa.array([True])) 