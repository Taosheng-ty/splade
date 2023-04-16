from string import Template
import random
import os
import time
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/..")
# experimentSettings={
#     # "Hard":"config.lambda_hard=10"  ,
#     "HardWithADWithAQ":"config.lambda_hard=10  config.lambda_Doc=10  config.lambda_Query=10"  ,
#     "HardWithADWithAQWithPsuedo":"config.lambda_hard=10  config.lambda_Doc=10  config.lambda_Query=10  config.lambda_psuedo=10"  ,
#     "HardWithAQ":"config.lambda_hard=10  config.lambda_Query=10"  ,
#     "HardWithAQWithPsuedo":"config.lambda_hard=10  config.lambda_Query=10  config.lambda_psuedo=10"  ,
#     "HardWithAD":"config.lambda_hard=10  config.lambda_Doc=10"  ,
#     "AQWADWPsuedo":"config.lambda_psuedo=10  config.lambda_Query=10  config.lambda_Doc=10"  ,
#     "AQWAD":"config.lambda_Doc=10  config.lambda_Query=10"  ,    
#     "onlyAQ":"config.lambda_Query=10  config.regularizer.FLOPS.lambda_q=0",    
#     "onlyAD":"config.lambda_Doc=10  config.regularizer.FLOPS.lambda_d=0 ",
#     "onlyPsuedo":"config.lambda_psuedo=10  config.regularizer.FLOPS.lambda_d=0",
#     "onlyADWPsuedo":"config.lambda_psuedo=10  config.lambda_Doc=10  config.regularizer.FLOPS.lambda_d=0",
#     "AQWPsuedo":"config.lambda_psuedo=10  config.lambda_Query=10"  ,
# }
experimentSettings={
    # "Hard":"config.lambda_hard=10"  ,
    "HardWithADWithAQ":"config.lambda_hard=10 config.lambda_Doc=10 config.lambda_Query=10",
    "HardWithADWithAQWithPsuedo":"config.lambda_hard=10 config.lambda_Doc=10 config.lambda_Query=10 config.lambda_psuedo=10",
    "HardWithAQ":"config.lambda_hard=10 config.lambda_Query=10",
    "HardWithAQWithPsuedo":"config.lambda_hard=10 config.lambda_Query=10 config.lambda_psuedo=10",
    "HardWithAD":"config.lambda_hard=10 config.lambda_Doc=10",
    
    "AQWADWPsuedo":"config.lambda_psuedo=10  config.lambda_Query=10  config.lambda_Doc=10"  ,
    "AQWAD":"config.lambda_Doc=10  config.lambda_Query=10"  ,    
    "onlyAQ":"config.lambda_Query=10"  ,    
    "onlyAD":"config.lambda_Doc=10"  ,
    "onlyADWPsuedo":"config.lambda_psuedo=10  config.lambda_Doc=10"  ,
    "AQWPsuedo":"config.lambda_psuedo=10  config.lambda_Query=10"  ,
    # "onlyPsuedo":"config.lambda_psuedo=10",
    # "InBatch":"config.inBatch=10"  ,
    # "InBatchWithAQADPsuedo":"config.inBatch=10  config.lambda_psuedo=10  config.lambda_Query=10  config.lambda_Doc=10"  ,
}

desc="full44G"
scriptTemplate="runNoGpuspecify.sh"
experimentsFolder="experiments/Apr16Retrain"


# desc="11GNoReg"
# scriptTemplate="runNoGpuspecify11GNotch.sh"
# experimentsFolder="experiments/Apr811GNoReg"

# desc="toy"
# scriptTemplate="runToy.sh"
# experimentsFolder="experiments/toy1kMixApr14Cos"
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