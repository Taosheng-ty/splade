from string import Template
import random
import os
import time
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/..")
# experimentSettings={
#     # "Hard":"config.lambda_hard=1",
#     "HardWithADWithAQ":"config.lambda_hard=1 config.lambda_Doc=1 config.lambda_Query=1",
#     "HardWithADWithAQWithPsuedo":"config.lambda_hard=1 config.lambda_Doc=1 config.lambda_Query=1 config.lambda_psuedo=1",
#     "HardWithAQ":"config.lambda_hard=1 config.lambda_Query=1",
#     "HardWithAQWithPsuedo":"config.lambda_hard=1 config.lambda_Query=1 config.lambda_psuedo=1",
#     "HardWithAD":"config.lambda_hard=1 config.lambda_Doc=1",
#     "AQWADWPsuedo":"config.lambda_psuedo=1 config.lambda_Query=1 config.lambda_Doc=1",
#     "AQWAD":"config.lambda_Doc=1 config.lambda_Query=1",    
#     "onlyAQ":"config.lambda_Query=1 config.regularizer.FLOPS.lambda_q=0",    
#     "onlyAD":"config.lambda_Doc=1 config.regularizer.FLOPS.lambda_d=0 ",
#     "onlyPsuedo":"config.lambda_psuedo=1 config.regularizer.FLOPS.lambda_d=0",
#     "onlyADWPsuedo":"config.lambda_psuedo=1 config.lambda_Doc=1 config.regularizer.FLOPS.lambda_d=0",
#     "AQWPsuedo":"config.lambda_psuedo=1 config.lambda_Query=1",
# }
experimentSettings={
    # "Hard":"config.lambda_hard=1",
    # "HardWithADWithAQ":"config.lambda_hard=1 config.lambda_Doc=1 config.lambda_Query=1",
    # "HardWithADWithAQWithPsuedo":"config.lambda_hard=1 config.lambda_Doc=1 config.lambda_Query=1 config.lambda_psuedo=1",
    # "HardWithAQ":"config.lambda_hard=1 config.lambda_Query=1",
    # "HardWithAQWithPsuedo":"config.lambda_hard=1 config.lambda_Query=1 config.lambda_psuedo=1",
    # "HardWithAD":"config.lambda_hard=1 config.lambda_Doc=1",
    "AQWADWPsuedo":"config.lambda_psuedo=1 config.lambda_Query=1 config.lambda_Doc=1",
    "AQWAD":"config.lambda_Doc=1 config.lambda_Query=1",    
    "onlyAQ":"config.lambda_Query=1",    
    "onlyAD":"config.lambda_Doc=1",
    "onlyPsuedo":"config.lambda_psuedo=1",
    "onlyADWPsuedo":"config.lambda_psuedo=1 config.lambda_Doc=1",
    "AQWPsuedo":"config.lambda_psuedo=1 config.lambda_Query=1",
    # "InBatch":"config.inBatch=1",
    # "InBatchWithAQADPsuedo":"config.inBatch=1 config.lambda_psuedo=1 config.lambda_Query=1 config.lambda_Doc=1",
}

# desc="fullDistill44G"
# scriptTemplate="runNoGpuspecify.sh"
# experimentsFolder="experiments/Apr8Noreg"


# desc="11GNoReg"
# scriptTemplate="runNoGpuspecify11GNotch.sh"
# experimentsFolder="experiments/Apr811GNoReg"

desc="toy"
scriptTemplate="runToyNanyuan.sh"
experimentsFolder="experiments/toy1kMix"
# scriptTemplate="runToyNanyuan.sh"
# scriptTemplate="run11GFullNanyuan.sh"
# desc="NanYuan11G"

os.makedirs(experimentsFolder, exist_ok=True)
for experimentName in experimentSettings:
    # Open the file
    experimentNameCur=experimentName+desc
    exprimentDir=f"{experimentsFolder}/{experimentNameCur}"
    with open(f'scripts/{scriptTemplate}', 'r') as file:
        # Read the contents of the file as a string
        file_contents = file.read()

        # Define the template string
        template = Template(file_contents)
        # Substitute placeholders within the string
        substituted_string = template.substitute(experimentsFolder=experimentsFolder,experimentName=experimentNameCur,Lossweight=experimentSettings[experimentName])

        # Print the substituted string
        # print(substituted_string)
    logPath=os.path.join(exprimentDir,"logs")
    os.makedirs(logPath, exist_ok=True)
    runShFile=os.path.join(logPath,"run.sh")
    with open(runShFile,"w") as file:
        file.write(substituted_string)
    print(f"sbatch {runShFile}")
    os.system(f"sbatch {runShFile}")
    # time.sleep(60)