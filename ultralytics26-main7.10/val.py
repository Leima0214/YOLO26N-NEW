
 
from ultralytics import YOLO

model = YOLO('best.pt')

 
# Train the model
results = model.val(data="coco128.yaml",  imgsz=640)


#详细的各类改进方法和流程操作，请关注B站博主：AI学术叫叫兽 
#详细的各类改进方法和流程操作，请关注B站博主：AI学术叫叫兽 
#详细的各类改进方法和流程操作，请关注B站博主：AI学术叫叫兽 