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
args = parser.parse_args()

# PyAudio setup
p=pyaudio.PyAudio()
FORMAT=pyaudio.paInt16
INPUT_CHANNELS  = 1
CHUNK=256
SIZE=8
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
        self.hold = 10
        self.threshold_high = 0.010
        self.threshold_low  = 0.010
        self.count = 0
        
    def connect(self,host,port):
        self.server.connect((host,port))

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

    def turn_right( self, diff ):
        self.wheel( self.handle + diff )

    def turn_left( self, diff ):
        self.wheel( self.handle - diff )


    def update( self, diff, volume ):
        # accel
        if( volume >= 0.15 ):
            self.accelerator( 30 )
            time.sleep(0.003)
            self.decelerator( 0)
            time.sleep(0.003)
        elif( volume >= 0.03 ):
            self.accelerator( 10 )
            time.sleep(0.003)
            self.decelerator( 0)
            time.sleep(0.003)
        else:
            self.accelerator( 0 )
            time.sleep(0.003)
            self.decelerator(100)
            time.sleep(0.003)

        # handling
        if( diff > 200 and volume >= self.threshold_high ):
            print("-->",volume)
            if self.handle < 50:
                self.turn_right(  5 )
                # self.wheel( 50 )
            else:
                self.turn_right(  3 )
        elif( diff < -100 and volume >= self.threshold_low ):
            print("<--",volume)
            if self.handle > 50:
                # self.wheel( 50 )
                self.turn_left(  5 )
            else:
                self.turn_left(  3 )
"""
        else: 
            if self.count < self.hold:
                self.count += 1
            else: # If pitch is not detected for some period.
                self.count = 0
                if self.handle > 50: # reverse to handle = 50
                    self.turn_left(2)
                elif self.handle < 50:
                    self.turn_right(2)
"""
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

def fft_and_filter( data ):
    windowed = np.hanning(len(data))*data
    result = np.abs(np.fft.fft(windowed))
    freq = np.fft.fftfreq(len(result), d = 1.0 / RATE)
    LOW_FREQ = 375.
    return result * (freq>=LOW_FREQ)

# initialize
car = Car()
wait = 0.05
print("connect to host =",args.host,"port =",args.port)
car.connect(args.host,args.port)
time.sleep(wait)
car.start()
time.sleep(wait)
car.accelerator(20)
time.sleep(wait)
car.decelerator(0)


data = np.zeros(CHUNK*SIZE)
SHIFT = 4

while input_stream.is_active():
    input = input_stream.read(CHUNK)
    max_val = 2.**15
    input = np.frombuffer(input, dtype=np.int16)/max_val
    prev_data = data.copy()
    data = np.concatenate( [ data[CHUNK:], input ] )
    
    v0 = fft_and_filter(prev_data)
    v1 = fft_and_filter(data)

    v_neg = np.dot( v0[SHIFT:-SHIFT], v1[:-2*SHIFT] )
    v_pos = np.dot( v0[SHIFT:-SHIFT], v1[2*SHIFT:] )

    # volume = np.sum(np.abs(input))
    volume = np.average(v1)
    diff = v_pos - v_neg
    car.update(diff,volume)
