# dora-gen72-test: 将reachy2的pick-place-dev迁移到了dora机器人上，主要改动如下：
1.在```node-hub```中添加```node-hub\dora-gen72```文件夹，该文件夹中包含了dora机器人的机械臂控制代码，以及奥比中光gemin335相机驱动。该文件夹主要用于代替```node-hub\dora-reachy2```
2.在```examples```文件夹中添加```examples\gen72```文件夹，该文件夹包含执行的yml文件。该文件夹主要用于代替```example\dora-reachy2```
3.在```node-hub\dora-object-to-pose```对于boxes2e的输入处理方式做了修改，可忽略。
4.删除了reachy-mobile-base,reachy-head节点，删除了image-right，image-left消息。将reachy2-righr-arm与reachy2-left-arm合并为一个节点```node-hub\dora-gen72\dora_gen72\arm.py```

## 系统硬件
控制器：笔记本  
机械臂：rml-gen72*2，left(192.168.1.18),right(192.168.1.19)   注意！！！机械臂控制器需要升级到最新版本，否则无法使用。 
夹爪：lebai(LMG-90)
相机：奥比中光gemin335

## 坐标系标定
注意！！！坐标系的标定决定了后续的代码是否正确，以及机械臂是否能够正确的抓取物体。

### 1.睿尔曼手眼标定
需要将相机检测到的物体坐标转换到机械臂base坐标下，所以需要进行摄像头与机械臂坐标关系的标定，并将标定后的旋转矩阵和平移矩替换中```examples\gen72\pick-place-gen72.py```文件中。左右机械臂都需要进行标定。具体标定方法见https://github.com/RealManRobot/hand_eye_calibration。可以使用```compute\hand_eye_calibration```进行标定

### 2.工具坐标系标定
在控制机械臂末端时需要将基于基坐标系控制的末端坐标系转换到工具坐标系下(夹爪坐标系)。具体标定教程见https://develop.realman-robotics.com/robot/teachingPendant/setting/。

## gemini335相机python驱动
在使用gemini335相机时，需要先安装gemin335相机的python驱动。具体安装方法见```compute\pyorbbecsdk```.

## 运行
如果需要修改机械臂的初始位置请在```examples\gen72\pick-place-gen72.py```文件中修改。



