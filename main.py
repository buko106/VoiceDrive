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
CHUNK=4000
RATE=44000
 
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
        if self.dec != param and 0 <= param <= 100:
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

    # mapping from voice to driving parameter
    def get_wheel( self, pitch ):
        diff = np.log(pitch/self.neutral) * 50
        return min(100,max(0,int(diff+50)))

    def get_accel_decel( self, volume ):
        threshold = 3.
        if( volume <= threshold ):
            return 0,20
        return min(50,int((volume-threshold)*4)),0

    # Called in while loop
    def update( self, pitch, volume ):
        w = self.get_wheel( pitch )
        a,d = self.get_accel_decel( volume )
        print(a,w)
        self.wheel(w)
        time.sleep(0.01)
        self.accelerator(a)
        time.sleep(0.01)
        self.decelerator(d)

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
    return data * np.exp(-freq/30)

import matplotlib.pyplot as plt
while input_stream.is_active():
    input = input_stream.read(CHUNK)
    input = np.frombuffer(input, dtype=np.int16)/max_val
    result,freq = fft_and_filter( input )
    low_pitch = accel_low_pitch(result,freq)
    pitch = freq[np.argmax(result)]
    volume = np.average(np.sort(result)[-1:])
    car.update( pitch, volume )
