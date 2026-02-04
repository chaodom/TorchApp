C++ libtorch实现各种模型推理

**已实现的模型：**
<pre>
分类
回归
crnn文字识别
检测（yolo11、yolo26、fasterRCNN）
分割（yolo11seg、yolo26seg、maskRCNN）
关键点检测（yolo11pose、yolo26pose、keypointRCNN）
</pre>
<br><br>
**继承关系：**  
Classify、Regress、CRNN类均直接继承自App  
YOLO11系列分支的继承关系如下（YOLO26同理）
<pre>
                    App                                         
                     |
                   Detect
         __________/ | \__________
        /            |            \
    Segment         Yolo        KeyPoint                  
       \   ________/ | \________   /
        \ /          |          \ /
     YoloSeg      Yolo11       YoloPose
          \__   ____/ \____   __/
             \ /           \ /
          Yolo11seg    Yolo11pose
               \____   ____/ 
                    \ /
                Yolo11Final
</pre>
RCNN系列分支的继承关系如下
<pre>
                App                                         
                 |
               Detect
     __________/ | \__________
    /            |            \
 Segment    FasterRCNN     KeyPoint                  
    \____   ____/ \____   ____/
         \ /           \ /
       MaskRCNN    KeyPointRCNN
           \____   ____/ 
                \ /            
             RCNNFinal
</pre>