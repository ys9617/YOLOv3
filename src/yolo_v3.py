import cv2
import numpy as np
import time

from openvino.inference_engine import IENetwork, IECore

labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


class Model_YOLO_V3:
    def __init__(self, model_name, device='CPU'):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.model = IECore().read_network(model=self.model_structure, weights=self.model_weights)
        
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_names = ['detector/yolo-v3/Conv_6/BiasAdd/YoloRegion', \
                             'detector/yolo-v3/Conv_14/BiasAdd/YoloRegion', \
                             'detector/yolo-v3/Conv_22/BiasAdd/YoloRegion']

        print('device : ', self.device)
        print('input name : ', self.input_name)
        print('input shape : ', self.input_shape)
        print('output names : ', self.output_names)


    def load_model(self):
        self.net = IECore().load_network(self.model, self.device)


    def predict(self, image):
        input_img = self.preprocess_input(image)

        start_time = time.time()

        output = self.net.infer({self.input_name:input_img})

        print('inference time : ', time.time() - start_time)

        return self.preprocess_output(output, image)


    def preprocess_input(self, image):
        p_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        p_img = cv2.resize(p_img, (self.input_shape[3], self.input_shape[2]))
        p_img = p_img.transpose((2,0,1))
        p_img = p_img.reshape(1, *p_img.shape)

        return p_img


    def preprocess_output(self, outputs, image):
        threshold = 0.6
        anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
        
        h, w, c = image.shape

        _, _, input_h, input_w = self.input_shape
        
        num_box = 3
        boxes = []

        for i in range(num_box):
            output = np.squeeze(outputs[self.output_names[i]])
            anchor = anchors[i]

            _, grid_h, grid_w = output.shape

            grid_width = input_w / grid_w
            grid_height = input_h / grid_h

            output = output.transpose((1,2,0))

            output = output.reshape((grid_h, grid_w, num_box, -1))
            
            for ih in range(grid_h):
                for iw in range(grid_w):
                    for ib in range(num_box):
                        objectness = output[ih][iw][ib][4]

                        if objectness >= threshold:
                            tx, ty, tw, th = output[ih][iw][ib][:4]

                            bx = round((iw + self.sigmoid(tx)) * grid_width) *  w / input_w
                            by = round((ih + self.sigmoid(ty)) * grid_height) *  h / input_h
                            bw = round(anchor[2*ib+0] * np.exp(tw)) * w / input_w
                            bh = round(anchor[2*ib+1] * np.exp(th)) * h / input_h

                            classes = output[ih][iw][ib][5:]
                            
                            boxes.append([bx,by,bh,bw, classes])


        nms_threshold = 0.4
        num_class = 80
        
        for c in range(num_class):
            sorted_idx = np.argsort([-box[4][c] for box in boxes])
            
            for i in range(len(sorted_idx)):
                idx_i = sorted_idx[i]

                if boxes[idx_i][4][c] != 0:
                    for j in range(i+1, len(sorted_idx)):
                        idx_j = sorted_idx[j]

                        if self.iou(boxes[idx_i], boxes[idx_j]) >= nms_threshold:
                            boxes[idx_j][4][c] = 0

        ret = []

        for box in boxes:
            for i in range(num_class):
                if box[4][i] > 0.5:
                    x0 = int(max(box[0]-box[3]/2, 0))
                    y0 = int(max(box[1]-box[2]/2, 0))
                    x1 = int(min(box[0]+box[3]/2, w-1))
                    y1 = int(min(box[1]+box[2]/2, h-1))

                    ret.append([x0, y0, x1, y1, labels[np.argmax(box[4])]])

        return ret


    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))


    def iou(self, box1, box2):
        box1_xmin = box1[0] - box1[3]/2
        box1_ymin = box1[1] - box1[2]/2
        box1_xmax = box1[0] + box1[3]/2
        box1_ymax = box1[1] + box1[2]/2

        box2_xmin = box2[0] - box2[3]/2
        box2_ymin = box2[1] - box2[2]/2
        box2_xmax = box2[0] + box2[3]/2
        box2_ymax = box2[1] + box2[2]/2

        intersect_w = self._interval_overlap([box1_xmin, box1_xmax], [box2_xmin, box2_xmax])
        intersect_h = self._interval_overlap([box1_ymin, box1_ymax], [box2_ymin, box2_ymax])
        intersect = intersect_w * intersect_h

        w1, h1 = box1_xmax-box1_xmin, box1_ymax-box1_ymin
        w2, h2 = box2_xmax-box2_xmin, box2_ymax-box2_ymin
        union = w1*h1 + w2*h2 - intersect

        return float(intersect) / union

    def _interval_overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2,x4) - x3
