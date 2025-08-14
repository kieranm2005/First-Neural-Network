import serial
import time

class FPGAAgent:
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200, input_scale=20):
        self.ser = serial.Serial(port, baudrate, timeout=0.1)
        self.input_scale = input_scale  # Scale obs[0] into a spike-driving input current

    def act(self, obs):
        current_input = int(obs[0] * self.input_scale)
        self.ser.write(bytes([current_input]))
        time.sleep(0.005)  # Small delay; tune this if needed
        response = self.ser.read(1)

        if response:
            spike = int.from_bytes(response, "little")
            return 2 if spike else 0  # e.g., spike = move forward, no spike = turn left
        else:
            return 0  # Default action

    def close(self):
        self.ser.close()
