from pynvml import *
nvmlInit()

deviceCount = nvmlDeviceGetCount()
print(deviceCount)

for i in range(deviceCount):
    handle = nvmlDeviceGetHandleByIndex(i)
    print("device {}: {}".format(i, nvmlDeviceGetName(handle)))