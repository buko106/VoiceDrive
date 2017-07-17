# -*- coding:utf-8 -*-

import pyaudio
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
CHUNK=2048
RATE=22000

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
    # TCP Util
    def connect(self,host,port):
        print("connect to host =",host,"port =",port)
        self.server.connect((host,port))
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
            self.server.sendall(b"shiftgear true\n" )
        else:
            self.server.sendall(b"shiftgear false\n" )
    # Wrapper
    def turn_right( self, diff ):
        self.wheel( self.handle + diff )

    def turn_left( self, diff ):
        self.wheel( self.handle - diff )

    # Called in while loop
    def update( self, diff, volume ):
        pass

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
"""
car = Car()
wait = 0.05
car.connect(args.host,args.port)
time.sleep(wait)
car.start()
time.sleep(wait)
car.accelerator(20)
time.sleep(wait)
car.decelerator(0)
"""

max_val = 2.**15
min_freq = 50
max_freq = 500
neutral  = args.freq

# voice utils
def fft_and_filter( data ):
    windowed = data * np.hanning(len(data))
    result   = np.abs(np.fft.fft(windowed))
    freq     = np.fft.fftfreq(len(result), d=1.0/RATE)
    idx      = np.logical_and(min_freq<=freq,freq<=max_freq)
    return result[idx], freq[idx]
import matplotlib.pyplot as plt
while input_stream.is_active():
    input = input_stream.read(CHUNK)
    input = np.frombuffer(input, dtype=np.int16)/max_val
    result,freq = fft_and_filter( input )
    print(freq[np.argsort(result)[-3:]])
