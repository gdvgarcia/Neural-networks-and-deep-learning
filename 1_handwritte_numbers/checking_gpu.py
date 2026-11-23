import torch

print("XPU Available:", torch.xpu.is_available())
if torch.xpu.is_available():
    print("Device Name:", torch.xpu.get_device_name(0))