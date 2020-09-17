# self-driving



- Lane Detection (Using OpenCV)
- Object Detection(Using yolo v3 tiny)

## Requirements
- Python 3.8.2
- OpenCV 4.2.0

<br>
<br>

## Inference
```python
python3 detect.py —cfg cfg/yolov3-tiny.cfg —weights yolov3-tiny.pt —source noshadow.mp4 
```
<br>
<br>


![main](https://user-images.githubusercontent.com/46425982/93428651-ce026f80-f8fa-11ea-8b4f-282732c06510.gif)

<br>
<br>
First, it works only for noshadow.mp4,<br>
It is difficult to detect lanes for challenge video.
<br>
<br>

I used yolov3 tiny because it is intended to be used on a raspberry pi.<br>
There is an improvement in performance when using the original yolov3.
