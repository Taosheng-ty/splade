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
# scriptTemplate="runNoGpuspecify.sh"
scriptTemplate="runToy.sh"
desc="toy-"
for experimentName in experimentSettings:
    # Open the file
    experimentNameCur=desc+experimentName
    exprimentDir=f"experiments/{experimentNameCur}"
    with open(f'scripts/{scriptTemplate}', 'r') as file:
        # Read the contents of the file as a string
        file_contents = file.read()

        # Define the template string
        template = Template(file_contents)
        # Substitute placeholders within the string
        substituted_string = template.substitute(experimentName=experimentNameCur,Lossweight=experimentSettings[experimentName])

        # Print the substituted string
        # print(substituted_string)
    logPath=os.path.join(exprimentDir,"logs")
    os.makedirs(logPath, exist_ok=True)
    runShFile=os.path.join(logPath,"run.sh")
    with open(runShFile,"w") as file:
        file.write(substituted_string)
    print(f"sbatch {runShFile}")
    os.system(f"sbatch {runShFile}")