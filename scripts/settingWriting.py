from string import Template
import random
import os
scriptPath=os.path.dirname(os.path.abspath(__file__))
os.chdir(scriptPath+"/..")
experimentSettings={
    "HardWithADWithAQ":"config.lambda_hard=1 config.lambda_Doc=1000 config.lambda_Query=1000",
    "HardWithADWithAQWithPsuedo":"config.lambda_hard=1 config.lambda_Doc=1000 config.lambda_Query=1000 config.lambda_psuedo=1000",
    "HardWithAD":"config.lambda_hard=1 config.lambda_Doc=1000",
    "HardWithAQ":"config.lambda_hard=1 config.lambda_Query=1000",
    "HardWithAQWithPsuedo":"config.lambda_hard=1 config.lambda_Query=1000 config.lambda_psuedo=1000",
    "ADWithAQ":"config.lambda_Query=1000 config.lambda_Doc=1000"
}

for experimentName in experimentSettings:
    # Open the file
    exprimentDir=f"experiments/{experimentName}"
    with open('scripts/runNoGpuspecify.sh', 'r') as file:
        # Read the contents of the file as a string
        file_contents = file.read()

        # Define the template string
        template = Template(file_contents)
        # Substitute placeholders within the string
        substituted_string = template.substitute(experimentName=experimentName,Lossweight=experimentSettings[experimentName])

        # Print the substituted string
        # print(substituted_string)
    logPath=os.path.join(exprimentDir,"logs")
    os.makedirs(logPath, exist_ok=True)
    runShFile=os.path.join(logPath,"run.sh")
    with open(runShFile,"w") as file:
        file.write(substituted_string)
    os.system(f"sbatch {runShFile}")