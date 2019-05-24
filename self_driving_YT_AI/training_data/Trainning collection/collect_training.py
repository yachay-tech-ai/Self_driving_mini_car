# USAGE
# python server.py  --montageW 2 --montageH 2

# import the necessary packages
from imutils import build_montages
from datetime import datetime
import numpy as np
import imagezmq
import argparse
import imutils
import cv2
import numpy as np
import cv2
import serial
import pygame
from pygame.locals import *
import socket
import time
import os
import struct
import pickle

class CollectTraining(object):

    def __init__(self,serial_port,input_size):

        self.ap = argparse.ArgumentParser()
        self.ap.add_argument("-mW", "--montageW", required=True, type=int,
            help="montage frame width")
        self.ap.add_argument("-mH", "--montageH", required=True, type=int,
            help="montage frame height")
        self.args = vars(self.ap.parse_args())

        # initialize the ImageHub object
        self.imageHub = imagezmq.ImageHub()

        self.lastActive = {}
        self.lastActiveCheck = datetime.now()

        # stores the estimated number of Pis, active checking period, and
        # calculates the duration seconds to wait before making a check to
        # see if a device was active
        self.ESTIMATED_NUM_PIS = 4
        self.ACTIVE_CHECK_PERIOD = 10
        self.ACTIVE_CHECK_SECONDS = self.ESTIMATED_NUM_PIS * self.ACTIVE_CHECK_PERIOD

        # assign montage width and height so we can view all incoming frames
        # in a single "dashboard"
        self.mW = self.args["montageW"]
        self.mH = self.args["montageH"]

        self.ser = serial.Serial(serial_port, 9600, timeout=1)
        self.send_inst = True
        self.input_size = input_size

        #labels
        self.k = np.zeros((4, 4), 'float')
        for i in range(4):
            self.k[i, i] = 1
        pygame.init()
        pygame.display.set_mode((250, 250))


    def collect(self):
        saved_frame = 0
        total_frame = 0

        # collect images for training
        print("Start collecting images...")
        print("Press 'q' or 'x' to finish...")
        start = cv2.getTickCount()

        X = np.empty((0, self.input_size))
        y = np.empty((0, 4))
        while self.send_inst:
            # receive RPi name and frame from the RPi and acknowledge
            # the receipt
            (rpiName, frame) = self.imageHub.recv_image()
            self.imageHub.send_reply(b'OK')

            if rpiName not in self.lastActive.keys():
                print("[INFO] receiving data from {}...".format(rpiName))

            # record the last active time for the device from which we just
            # received a frame
            self.lastActive[rpiName] = datetime.now()

            # resize the frame to have a maximum width of 400 pixels, then
            # grab the frame dimensions and construct a blob
            frame = imutils.resize(frame, width=320)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (320, 240)),
                0.007843, (320, 240), 127.5)
            blob = np.squeeze(blob)
            cv2.imshow('Image',frame)
            print(blob.shape)
            height, width = blob.shape
            roi = blob[int(height/2):height, :]
            temp_array = roi.reshape(1, int(height/2) * width).astype(np.float32)
            frame += 1
            total_frame += 1                
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    key_input = pygame.key.get_pressed()

                    # simple orders
                    if key_input[pygame.K_UP]:
                        print("Forward")
                        saved_frame += 1
                        X = np.vstack((X, temp_array))
                        y = np.vstack((y, self.k[2]))
                        self.ser.write(chr(6).encode())

                    elif key_input[pygame.K_DOWN]:
                        print("Reverse")
                        self.ser.write(chr(7).encode())

                    elif key_input[pygame.K_RIGHT]:
                        print("Right")
                        X = np.vstack((X, temp_array))
                        y = np.vstack((y, self.k[1]))
                        saved_frame += 1
                        self.ser.write(chr(3).encode())

                    elif key_input[pygame.K_LEFT]:
                        print("Left")
                        X = np.vstack((X, temp_array))
                        y = np.vstack((y, self.k[0]))
                        saved_frame += 1
                        self.ser.write(chr(4).encode())

                    elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                        print("exit")
                        self.send_inst = False
                        self.ser.write(chr(0).encode())
                        self.ser.close()
                        break

                elif event.type == pygame.KEYUP:
                    self.ser.write(chr(0).encode())


            # detect any kepresses
            key = cv2.waitKey(1) & 0xFF

            # if current time *minus* last time when the active device check
            # was made is greater than the threshold set then do a check
            if (datetime.now() - self.lastActiveCheck).seconds > self.ACTIVE_CHECK_SECONDS:
                # loop over all previously active devices
                for (rpiName, ts) in list(self.lastActive.items()):
                    # remove the RPi from the last active and frame
                    # dictionaries if the device hasn't been active recently
                    if (datetime.now() - ts).seconds > self.ACTIVE_CHECK_SECONDS:
                        print("[INFO] lost connection to {}".format(rpiName))
                        self.lastActive.pop(rpiName)
                        

                # set the last active check time as current time
                self.lastActiveCheck = datetime.now()

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        # save data as a numpy file
        file_name = str(int(time.time()))
        directory = "training_data"
        if not os.path.exists(directory):
            os.makedirs(directory)
        try:
            np.savez(directory + '/' + file_name + '.npz', train=X, train_labels=y)
        except IOError as e:
            print(e)

        end = cv2.getTickCount()
        # calculate streaming duration
        print("Streaming duration: , %.2fs" % ((end - start) / cv2.getTickFrequency()))

        print(X.shape)
        print(y.shape)
        print("Total frame: ", total_frame)
        print("Saved frame: ", saved_frame)
        print("Dropped frame: ", total_frame - saved_frame)

       

        # do a bit of cleanup
        cv2.destroyAllWindows()


# serial port
sp = "COM8"

# vector size, half of the image
s = 120 * 320

ctd = CollectTraining(sp, s)
ctd.collect()