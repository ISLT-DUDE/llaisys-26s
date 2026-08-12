import llaisys
print("DeviceType members:")
for d in llaisys.DeviceType:
    print(f"  {d.name} = {d.value}")