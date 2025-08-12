import serial
import time

class FPGAAgent:
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=1)

    def act(self, obs):
        obs_byte = bytes([1]) if obs[0] > 0.5 else bytes([0])
        self.ser.write(obs_byte)
        time.sleep(0.01)  # wait a bit for FPGA response
        response = self.ser.read(1)
        if response:
            spike = int.from_bytes(response, "little")
            return spike % 3  # Map spike to action
        else:
            return 0  # default to turning left

    def close(self):
        self.ser.close()
