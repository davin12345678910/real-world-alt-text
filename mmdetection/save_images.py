from mmdeploy_python import Detector
import cv2
import sys

def check_knife(img):
    img = cv2.imread('' + img)

    # create a detector
    detector = Detector(model_path='work_dirs/rtmdet-sdk', device_name='cuda', device_id=0)
    # run the inference
    bboxes, labels, _ = detector(img)
    # Filter the result according to threshold
    indices = [i for i in range(len(bboxes))]
    for index, bbox, label_id in zip(indices, bboxes, labels):
        [left, top, right, bottom], score = bbox[0:4].astype(int),  bbox[4]
        print(label_id, labels)
        if score < 0.3:
            continue
        # draw bbox
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))

    cv2.imwrite('output_detection.png', img)


def main():
    args = sys.argv
    check_knife(args[1])

if __name__ == '__main__':
    main()