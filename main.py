# -*- coding:utf-8 -*-

import pyaudio
import pygame
import numpy as np
import socket
import time
import argparse

description="Driving Interface using Voice Pitch Detection."
parser = argparse.ArgumentParser(description=description)
parser.add_argument("-p","--port", type=int, default=5800)
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("-f","--freq", type=float, default=220)
args = parser.parse_args()

# PyAudio setup
p=pyaudio.PyAudio()
FORMAT=pyaudio.paInt16
INPUT_CHANNELS  = 1
RATE=20000
CHUNK=RATE//10

# 
input_stream=p.open( format = FORMAT,
		     channels = INPUT_CHANNELS,
		     rate = RATE,
		     frames_per_buffer = CHUNK,
		     input = True)


# Driving Class
class Car:
    def __init__(self):
        self.acc = 0
        self.dec = 0
        self.gear = True
        self.handle = 50
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.neutral = args.freq
        self.flag = 0
        self.arrow = np.array([(20,0),(20,-150),(40,-150),(0,-200),(-40,-150),(-20,-150),(-20,0)],dtype=np.float)
        self.acc_max = 25

    # TCP Util
    def connect(self,host,port):
        print("connect to host =",host,"port =",port)
        self.server.connect((host,port))
    # GUI Util
    def create_window(self):
        pygame.init()
        self.screen = pygame.display.set_mode((500,500))
        self.screen.fill((0,0,0))
        pygame.display.set_caption("VoiceDrive")
        
    # Driving Utils
    def start(self):
        self.server.sendall(b"start\n")
    def accelerator( self, param ):
        if 0 <= param <= 100:
            self.acc = param
            self.server.sendall(b"accelerator %d\n" % param )
    def decelerator( self, param ):
        if 0 <= param <= 100:
            self.dec = param
            self.server.sendall(b"decelerator %d\n" % param )
    def wheel( self, param ):
        if 0 <= param <= 100:
            self.handle = param
            self.server.sendall(b"wheel %d\n" % param )
    def shiftgear( self, param ):
        self.gear = param
        if self.gear:
            self.server.sendall(b"shiftgear 1\n" )
        else:
            self.server.sendall(b"shiftgear -1\n" )
    # Wrapper
    def turn_right( self, diff ):
        self.wheel( self.handle + diff )

    def turn_left( self, diff ):
        self.wheel( self.handle - diff )

    # mapping from voice to driving parameter
    def get_wheel( self, pitch ):
        diff = np.log(pitch/self.neutral) * 50
        return min(100,max(0,int(diff+50)))

    def get_accel_decel( self, volume ):
        threshold = 3.
        if( volume <= threshold ):
            return 0,50
        return min(self.acc_max,int((volume-threshold)*2)),0
    def update_window( self ):
        self.screen.fill((0,0,0))
        theta = np.radians(self.handle-50)*1.4
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        if self.gear :
            arrow_polygon = (R.dot(self.arrow.T).T)*(0.1+0.9*self.acc/self.acc_max) + np.array([250,250])
        else:
            arrow_polygon = (R.dot(self.arrow.T).T)*(0.1+0.9*self.acc/self.acc_max)*np.array([[1,-1]]) + np.array([250,50])
        pygame.draw.polygon(self.screen,(255,255,0),arrow_polygon)
        pygame.display.update()
    # Called in while loop
    def update( self, pitch, volume ):
        w = self.get_wheel( pitch )
        a,d = self.get_accel_decel( volume )

        some_key_pressed = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.shiftgear( not self.gear )
                    print("gear =",self.gear)
                    some_key_pressed = True
                if event.key == pygame.K_ESCAPE:
                    self.start()
                    print("start",self.gear)
                    some_key_pressed = True
                    
        if not some_key_pressed:
            if self.flag%3 == 0:
                if d == 0:
                    self.wheel(w)
            elif self.flag%3 == 1:
                self.accelerator(a)
            elif self.flag%3 == 2:
                self.decelerator(d)
            self.flag += 1

        print(self.acc,self.dec,self.handle,self.gear)
        self.update_window()
############################################################
# 初期状態に戻す 計測開始
# start 
# アクセルレベルの設定 (0 ... 100)　更新されるまで同じレベルを維持する。
# accelerator 50
# ブレーキレベルの設定 (0 ... 100)　更新されるまで同じレベルを維持する。
# decelerator 50
# ハンドルレベル　(0 ... 100) 更新されるまで同じレベルを維持する 50がニュートラル
# wheel 50
# ギア true=前進 false=後退 
# shiftgear true
############################################################


# initialize
car = Car()
car.create_window()
wait = 0.05
car.connect(args.host,args.port)
time.sleep(wait)
car.start()
time.sleep(wait)
car.accelerator(0)
time.sleep(wait)
car.decelerator(0)
time.sleep(wait)

max_val = 2.**15
min_freq = 100
max_freq = 500

# voice utils
def fft_and_filter( data ):
    windowed = data * np.hanning(len(data))
    result   = np.abs(np.fft.fft(windowed))
    freq     = np.fft.fftfreq(len(result), d=1.0/RATE)
    idx      = np.logical_and(min_freq<=freq,freq<=max_freq)
    return result[idx], freq[idx]

def accel_low_pitch( data, freq ):
    return data * freq**-2

import matplotlib.pyplot as plt
while input_stream.is_active():
    input = input_stream.read(CHUNK)
    input = np.frombuffer(input, dtype=np.int16)/max_val
    result,freq = fft_and_filter( input )
    low_pitch = accel_low_pitch(result,freq)
    max_idx = np.argmax(low_pitch)
    pitch = freq[max_idx]
    volume = np.average(np.sort(result)[-1:])
    car.update( pitch, volume )
