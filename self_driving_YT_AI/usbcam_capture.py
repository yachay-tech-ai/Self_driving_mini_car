from time import sleep
import cv2
import pygame
from pygame.locals import *
import time

pygame.init()
pygame.display.set_mode((250, 250))

time.sleep(0.1)

frame_rate_calc = 1
freq = cv2.getTickFrequency()
cap = cv2.VideoCapture(0)


# Camera warm-up time
sleep(2)

while True:

    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow('image',img)
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            key_input = pygame.key.get_pressed()     
            if key_input[pygame.K_UP]:
                print("key up pressed. Captured")
                cv2.imwrite(str(time.time())+'.jpg',img)
            elif key_input[pygame.k_DOWN]:
                break
