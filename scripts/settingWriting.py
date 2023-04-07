from string import Template
import random
import os
import time
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/..")
experimentSettings={
    # "Hard":"config.lambda_hard=1",
    # "HardWithADWithAQ":"config.lambda_hard=1 config.lambda_Doc=1000 config.lambda_Query=1000",
    # "HardWithADWithAQWithPsuedo":"config.lambda_hard=1 config.lambda_Doc=1000 config.lambda_Query=1000 config.lambda_psuedo=1000",
    # "HardWithAQ":"config.lambda_hard=1 config.lambda_Query=1000",
    # "HardWithAQWithPsuedo":"config.lambda_hard=1 config.lambda_Query=1000 config.lambda_psuedo=1000",
    # "AQWADWPsuedo":"config.lambda_psuedo=1000 config.lambda_Query=1000 config.lambda_Doc=1000",
    # "AQWAD":"config.lambda_Doc=1000 config.lambda_Query=1000",    
    "onlyAQ":"config.lambda_Query=1000 config.regularizer.FLOPS.lambda_q=0.0001 config.regularizer.FLOPS.lambda_d=0.001",    
    # "HardWithAD":"config.lambda_hard=1 config.lambda_Doc=1000",
    # "onlyAD":"config.lambda_Doc=1000",
    # "onlyPsuedo":"config.lambda_psuedo=1000",
    # "onlyADWPsuedo":"config.lambda_psuedo=1000 config.lambda_Doc=1000",
    # "AQWPsuedo":"config.lambda_psuedo=1000 config.lambda_Query=1000",
}
# desc="fullDistill44G"
# scriptTemplate="runNoGpuspecify.sh"


desc="toy"
scriptTemplate="runToy.sh"
experimentsFolder="experiments/toy/"
os.makedirs(experimentsFolder, exist_ok=True)
scriptTemplate="runToyNanyuan.sh"
scriptTemplate="run11GFullNanyuan.sh"
desc="NanYuan11G"
for experimentName in experimentSettings:
    # Open the file
    experimentNameCur=desc+experimentName
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