import cv2 as cv2
import os, time
import argparse
import numpy as np

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))


def parse_flow():
    parser = argparse.ArgumentParser(description='Optical flow')
    parser.add_argument('--dataset', type=str, default='video_human/',
                        help='dataset name (default: video_human)')
    parser.add_argument('--method', type=str, default='Farneback',
                        help='optical flow method (default: Farneback)')
    # the parser
    args = parser.parse_args()
    return args

class FLOW():
    def __init__(self):
        print('FLOW class init')
        self.hsv = []
        self.mask = []

    def get_data(self, folder):
        img_paths = []
        for root, _, files in os.walk(folder):
            for filename in files:
                if filename.endswith(".png"):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    img_paths.append(imgpath)
        print('Found {} images in the folder {}'.format(len(img_paths), folder))
        img_paths.sort()
        return img_paths

    def farneback(self, old_gray, new_gray):
        flow = cv2.calcOpticalFlowFarneback(old_gray,new_gray, None, 0.5,3,15,3,5,1.1,0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        self.hsv[...,0] = ang*180/np.pi/2
        self.hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(self.hsv,cv2.COLOR_HSV2BGR)

        #cv2.imshow('frame2',rgb)
        return rgb

    def lucaskanade(self, old_gray, new, p0):    
        new_gray = cv2.cvtColor(new,cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

        # Select good points
        print(p1[st==1])
        good_news = p1[st==1]
        good_olds = p0[st==1]

        # draw the tracks
        for i,(good_new,good_old) in enumerate(zip(good_news,good_olds)):
            a,b = good_new.ravel()
            c,d = good_old.ravel()
            self.mask = cv2.line(self.mask, (a,b),(c,d), color[i].tolist(), 2)
            new = cv2.circle(new,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(new,self.mask)

        return img, good_news
        

    def main(self, root, method):
        img_paths = self.get_data(root)
        for idx, item in enumerate(img_paths):
            new = cv2.imread(item)
            if idx == 0:
                if method == 'Farneback':
                    self.hsv = np.zeros_like(new)
                    self.hsv[...,1] = 255
                    old_gray = cv2.cvtColor(new,cv2.COLOR_BGR2GRAY)
                elif method == 'lucaskanade':
                    old_gray = cv2.cvtColor(new,cv2.COLOR_BGR2GRAY)
                    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
                    self.mask = np.zeros_like(new)
                else:
                    print("Input wrong method")
                    return
            else:
                if method == 'Farneback':
                    new_gray = cv2.cvtColor(new,cv2.COLOR_BGR2GRAY)
                    rgb = self.farneback(old_gray,new_gray)
                    old = new
                elif method == 'lucaskanade':
                    rgb, good_new = self.lucaskanade(old_gray,new, p0)
                    old = new
                    p0 = good_new.reshape(-1,1,2)

                cv2.imshow('frame2',rgb)
                k = cv2.waitKey(1) & 0xff 
                if k == 27:
                    break
                # cv2.imwrite('./test_result/opti_{:04d}.png'.format(idx), rgb)
                print('[{:04d}/{:04d}] Optical Flow image'.format(idx, len(img_paths)-1))
            
            
if __name__ == "__main__":
    root = '/home/hosun/'
    arg = parse_flow()
    flow = FLOW()
    img_root = os.path.join(root, arg.dataset)
    flow.main(img_root, arg.method)