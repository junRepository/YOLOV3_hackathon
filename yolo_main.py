import cv2
import pafy
import numpy as np
import time # -- 프레임 계산을 위해 사용
import pandas as pd
# import youtube_frame.py as yf

video_path = '1.mp4' #-- 사용할 영상 경로
model_file = './yolov3.weights' #-- 본인 개발 환경에 맞게 변경할 것
config_file = './yolov3.cfg' #-- 본인 개발 환경에 맞게 변경할 것
min_confidence = 0.5
window_x = 960
window_y = 540

def detectAndDisplay(frame):
    in_count = 0
    out_count = 0
    people_count = []
    

    start_time = time.time()
    img = cv2.resize(frame, (window_x, window_y)) # 화면 사이즈 강제 조절
    # img = cv2.resize(frame, None, fx=0.5, fy=0.5)
    height, width, channels = img.shape
    #cv2.imshow("Original Image", img)

    #--cnn관련 
    blob = cv2.dnn.blobFromImage(img, 0.00392, (224, 224), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    #-- 탐지한 객체의 클래스 예측 
    class_ids = []
    confidences = []
    boxes = []
    coordinate = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            #-- 원하는 class id 입력/ coco.names의 id에서 -1 할 것 
            if class_id == 0 and confidence > min_confidence:
                #-- 탐지한 객체 박싱
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
               
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    font = cv2.FONT_HERSHEY_DUPLEX
    
    # ##프레임기반 인원 수 카운트
    # people_count.append([len(indexes)])
    # # pe = [x for x in people_count]
    # print(people_count)
    

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = "{}: {:.2f}".format(classes[class_ids[i]], confidences[i]*100)
            # print(i, label)
            color = (0,0,255)#colors[i] #-- 경계 상자 컬러 설정 / 단일 생상 사용시 (255,255,255)사용(B,G,R)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)        
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)

            coordinate.append([x,y,x+w,y+h])
    #들어오는 사람들과 나가는 사람들 수 count
    # for i in range(len(coordinate)):
    #     a = coordinate[i][0]
    #     b = coordinate[i][1]
    #     c = coordinate[i][2]
    #     d = coordinate[i][3]
    #     print(a,b,c,d)
    #     if a>10 and b>10 and c<950 and d<530:
    #         in_count += 1 
    #     elif a<20 or b<20 or c>900 or d>500:
    #         out_count += 1
    # print(in_count, out_count)

    end_time = time.time()
    process_time = end_time - start_time
    # print("=== A frame took {:.3f} seconds".format(process_time))
    cv2.imshow("YOLO test", img)


net = cv2.dnn.readNet(model_file, config_file)
#-- GPU 사용
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = []
with open("./yolo.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#-- 비디오 활성화
cap = cv2.VideoCapture(video_path) #-- 웹캠 사용시 vedio_path를 0 으로 변경

if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    #-- q 입력시 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# def count(coordinate):
#     in_count = 0
#     out_count = 0
#     for i in range(len(coordinate)):
#         a = coordinate[i][0]
#         b = coordinate[i][1]
#         c = coordinate[i][2]
#         d = coordinate[i][3]
#         print(a,b,c,d)
#         if a>10 and b>10 and c<950 and d<530:
#             in_count += 1 
#         elif a<20 or b<20 or c>900 or d>500:
#             out_count += 1
#     print(in_count, out_count)