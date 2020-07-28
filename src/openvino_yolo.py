import argparse
import cv2
import numpy as np
from matplotlib import pyplot
from matplotlib.patches import Rectangle

from yolo_v3 import Model_YOLO_V3 as YOLO


def main(args):
    device = args.device
    input_type = args.input_type
    input_file = args.input_file

    # OpenVINO path
    model = '../models/yolov3'

    net = YOLO(model, device)
    net.load_model()

    if input_type == 'image':
        img = cv2.imread(input_file)
        outputs = net.predict(img)

        for box in outputs:
            x0 = box[0]
            y0 = box[1]
            x1 = box[2]
            y1 = box[3]
            label = box[4]

            cv2.rectangle(img, (x0, y0),  (x1, y1), (0, 0, 255), 1)
            cv2.putText(img, label, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        
        cv2.imshow('img', img)
        cv2.waitKey(0)
    elif input_type == 'video' or input_type == 'cam':
        if input_type == 'cam':
            input_file = 0
            
        cap = cv2.VideoCapture(input_file)

        w = int(cap.get(3))
        h = int(cap.get(4))

        while cap.isOpened():
            flag, frame = cap.read()

            if not flag:
                break


            outputs = net.predict(frame)
            
            for box in outputs:
                x0 = box[0]
                y0 = box[1]
                x1 = box[2]
                y1 = box[3]
                label = box[4]

                cv2.rectangle(frame, (x0, y0),  (x1, y1), (0, 0, 255), 1)
                cv2.putText(frame, label, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

            cv2.imshow('frame', frame)


            key_pressed = cv2.waitKey(60)

            if key_pressed == 27:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='CPU')
    parser.add_argument('--input_type', default='image')
    parser.add_argument('--input_file', default='')

    args = parser.parse_args()

    main(args)








