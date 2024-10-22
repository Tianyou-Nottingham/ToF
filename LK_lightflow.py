import cv2
import numpy as np
import serial
import configs.config as cfg


lk_params = dict(winSize = (80, 80),
                 maxLevel = 5,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners = 100,
                      qualityLevel = 0.3,
                      minDistance = 80,
                      blockSize = 7)

class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 10
        self.tracks = []
        self.frame_idx = 0
        self.cam = cv2.VideoCapture(video_src)

    def run(self):
        while True:
            ret, frame = self.cam.read()
            if ret == True:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                vis = frame.copy()

                if len(self.tracks) > 0:
                    img0, img1 = self.prev_gray, frame_gray
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                    d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                    good = d < 1
                    new_tracks = []
                    for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                        if not good_flag:
                            continue
                        tr.append((x, y))
                        if len(tr) > self.track_len:
                            del tr[0]
                        new_tracks.append(tr)
                        cv2.circle(vis, (round(x), round(y)), 5, (0, 255, 0), -1)
                    self.tracks = new_tracks
                    cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

                if self.frame_idx % self.detect_interval == 0:
                    mask = np.zeros_like(frame_gray)
                    mask[:] = 255
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                        cv2.circle(mask, (x, y), 5, 0, -1)

                    dst = cv2.cornerHarris(frame_gray, 20, 3, 0.04)
                    p = np.where(dst > 0.01 * dst.max())
                    # p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            self.tracks.append([(x, y)])

                self.frame_idx += 1
                self.prev_gray = frame_gray
                cv2.imshow('lk_track', vis)
                cv2.waitKey(200)

                ch = cv2.waitKey(1) & 0xFF
                if ch == 27:
                    break

def main():
    import sys
    try: video_src = sys.argv[1]
    except: video_src = "./output.avi"
    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
