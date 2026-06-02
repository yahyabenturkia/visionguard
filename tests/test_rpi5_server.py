import socket
import time

s = socket.socket()
s.connect(('192.168.10.2', 9999))
print("Connected to RPi5 server")

s.sendall(b'CAPTURE:good\n')
response = s.recv(1024).decode().strip()
print(f"CAPTURE:good → {response}")

time.sleep(1)

s.sendall(b'CAPTURE:defect\n')
response = s.recv(1024).decode().strip()
print(f"CAPTURE:defect → {response}")

time.sleep(1)

s.sendall(b'QUIT\n')
response = s.recv(1024).decode().strip()
print(f"QUIT → {response}")

s.close()