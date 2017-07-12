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

# TCP Connection
host = args.host
port = args.port
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

# Driving utils
def start(server=server):
    server.sendall(b"start\n")
def accelerator( param, server=server ):
    server.sendall(b"accelerator %d\n" % param )
def decelerator( param, server=server ):
    server.sendall(b"decelerator %d\n" % param )
def wheel( param, server=server ):
    server.sendall(b"wheel %d\n" % param )
def shiftgear( param, server=server ):
    if param:
        server.sendall(b"shiftgear true\n" % param )
    else:
        server.sendall(b"shiftgear false\n" % param )

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
"""
# initialize
server.connect((host, port))
start()
time.sleep(0.01)
accelerator(20)
time.sleep(0.01)
decelerator(0)
"""

def fft_and_filter( data ):
    windowed = np.hanning(len(data))*data
    result = np.abs(np.fft.fft(windowed))
    LOW_FREQ = 375.
    freq = np.fft.fftfreq(len(result), d = 1.0 / RATE)
    return result * (freq>=LOW_FREQ)
    

data = np.zeros(CHUNK*SIZE)
LOW_FREQ = 375.
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

    val = v_pos - v_neg
    if np.abs(val) > 50:
        print(val)
