import torch
import cv2


class WeaponDetector:
    def __init__(self, path_to_weights):
        self.model = torch.hub.load('yolov5', 'custom', path_to_weights, source='local')
        self.model.conf = 0.25
        self.model.iou = 0.45
        self.model.img = 640

    def detect(self, capture, every_kth_frame: int):
        frame_index = -1
        results = []
        while True:
            print(frame_index)
            ret, frame = capture.read()
            if not ret:
                break
            frame_index += 1
            if frame_index % every_kth_frame == 0:
                continue
            results.append(self.model(frame, size=640))

        results.sort(key=lambda x: x.pandas().xyxy[0]['confidence'].max(), reverse=True)
        results = results[:5]
        if 0 < len(results) < 5:
            results += [results[-1]]*(5 - len(results))

        for i in range(len(results)):
            results[i].save(save_dir = 'data/output/images')


if __name__ == '__main__':
    weapon_detector = WeaponDetector('weights/best.pt')
    cap = cv2.VideoCapture(r"C:\Users\79787\Downloads\video.mp4")

    weapon_detector.detect(cap, 300)

