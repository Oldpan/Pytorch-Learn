import torch
from tqdm import tqdm
import cv2
import json
import numpy as np
import sys
import time
import torch.multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Queue as pQueue
from threading import Thread

# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    from queue import Queue, LifoQueue
# otherwise, import the Queue class for Python 2.7
else:
    from Queue import Queue, LifoQueue


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas


def prep_frame(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))        # resize image with ratio unchanged
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


class WebcamLoader:
    def __init__(self, webcam, batchSize=1, queueSize=256):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(int(webcam))
        assert self.stream.isOpened(), 'Cannot capture source'
        self.stopped = False
        # initialize the queue used to store frames read from
        # the video file
        self.batchSize = batchSize
        self.Q = LifoQueue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        i = 0
        while True:
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                img = []
                orig_img = []
                im_name = []
                im_dim_list = []
                for k in range(self.batchSize):
                    (grabbed, frame) = self.stream.read()
                    # if the `grabbed` boolean is `False`, then we have
                    # reached the end of the video file
                    if not grabbed:
                        self.stop()
                        return
                    inp_dim = int(224)
                    img_k, orig_img_k, im_dim_list_k = prep_frame(frame, inp_dim)

                    img.append(img_k)
                    orig_img.append(orig_img_k)
                    im_name.append(str(i) + '.jpg')
                    im_dim_list.append(im_dim_list_k)

                with torch.no_grad():
                    # Human Detection
                    img = torch.cat(img)
                    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

                    self.Q.put((img, orig_img, im_name, im_dim_list))
                    i = i + 1

            else:
                with self.Q.mutex:
                    self.Q.queue.clear()

    def videoinfo(self):
        # indicate the video info
        fourcc = int(self.stream.get(cv2.CAP_PROP_FOURCC))
        fps = self.stream.get(cv2.CAP_PROP_FPS)
        frameSize = (int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return fourcc, fps, frameSize

    def getitem(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue size
        return self.Q.qsize()

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


class GestureLoader:
    def __init__(self, loader, model, batchSize=1, queueSize=1024):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.model = model
        self.loader = loader
        self.model.cuda()
        self.model.eval()

        self.stopped = False
        self.batchSize = batchSize
        # initialize the queue used to store frames read from
        # the video file
        self.Q = LifoQueue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping the whole dataset
        while True:
            img, orig_img, im_name, im_dim_list = self.loader.getitem()

            with self.loader.Q.mutex:
                self.loader.Q.queue.clear()
            with torch.no_grad():
                img = img.cuda()
                # self.model.half()
                # prediction = self.model(img.half())
                prediction = self.model(img)
                for i in range(len(prediction)):
                    while self.Q.full():
                        time.sleep(0.2)
                    self.Q.put((prediction[i], orig_img[i], im_name[i]))

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue len
        return self.Q.qsize()


class DataWriter:
    def __init__(self, save_video=False,
                 savepath='examples/1.avi', fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=25, frameSize=(640, 480),
                 queueSize=1024):
        if save_video:
            # initialize the file video stream along with the boolean
            # used to indicate if the thread should be stopped or not
            self.stream = cv2.VideoWriter(savepath, fourcc, fps, frameSize)
            assert self.stream.isOpened(), 'Cannot open video for writing'
        self.stone = cv2.imread('/home/prototype/Documents/gesture/show/rock.png')
        self.paper = cv2.imread('/home/prototype/Documents/gesture/show/paper.png')
        self.scissors = cv2.imread('/home/prototype/Documents/gesture/show/scissors.png')
        self.save_video = save_video
        self.stopped = False
        self.final_result = []
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)
        # if opt.save_img:
        #     if not os.path.exists(opt.outputpath + '/vis'):
        #         os.mkdir(opt.outputpath + '/vis')

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                if self.save_video:
                    self.stream.release()
                return
            # otherwise, ensure the queue is not empty
            if not self.Q.empty():
                predict, orig_img, im_name = self.Q.get()
                gesture = predict.argmax()
                if gesture.item() == 0:
                    cv2.putText(orig_img, 'paper', (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2)
                    cv2.imshow("Gesture Demo", orig_img)
                    cv2.imshow("Game", self.scissors)
                elif gesture.item() == 1:
                    cv2.putText(orig_img, 'scissor', (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2)
                    cv2.imshow("Gesture Demo", orig_img)
                    cv2.imshow("Game", self.stone)
                else:
                    cv2.putText(orig_img, 'stone', (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2)
                    cv2.imshow("Gesture Demo", orig_img)
                    cv2.imshow("Game", self.paper)
                cv2.waitKey(30)
            else:
                time.sleep(0.1)

    def running(self):
        # indicate that the thread is still running
        time.sleep(0.2)
        return not self.Q.empty()

    def save(self, predict, orig_img, im_name):
        # save next frame in the queue
        self.Q.put((predict, orig_img, im_name))

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        time.sleep(0.2)

    def results(self):
        # return final result
        return self.final_result

    def len(self):
        # return queue len
        return self.Q.qsize()
