import subprocess
from collections import defaultdict
batcmd="sinfo -N -p notchpeak-gpu-guest --Format='nodehost,gres:50,gresused:50'"
result = subprocess.check_output(batcmd, shell=True)
# print(result.splitlines())

AvailableGPU=defaultdict(int)
AvailableGPUdetails=defaultdict(dict)
for lines in result.splitlines():
    lines=lines.decode("utf-8") 
    # print(lines)
    node,total,used=lines.split()
    # print(node,total,used)
    if "gpu:" in total:
        gpu=total.split(":")[1]
        left=int(total.split("(")[0].split(":")[2])-int(used.split("(")[0].split(":")[2])
        AvailableGPU[gpu]+=left
print(AvailableGPU)
    
    