import shutil
import sys
import subprocess

# Define source and target file paths
source_dir = "SecLMM/based_on_tinyclip/src/training/model/"
target_file = "/home/na/miniconda3/envs/seclmm/lib/python3.9/site-packages/transformers/models/clip/modeling_flax_clip.py"

# Parse command-line arguments
args = sys.argv[1:]  
task_name = None
hidden_act = None
softmax_act = None

# Extract arguments for task name, hidden activation, and softmax activation
for i in range(len(args)):
    if args[i] == "--task_name":
        task_name = args[i + 1]
    elif args[i] == "--hidden_act":
        hidden_act = args[i + 1]
    elif args[i] == "--softmax_act":
        softmax_act = args[i + 1]

# Validate the task name
valid_task_names = ["cifar10", "cifar100", "tiny-imagenet", "fairface-age", "fairface-gender", "fairface-race"]
if task_name not in valid_task_names:
    print(f"Invalid tasks: {', '.join(valid_task_names)}。")
    sys.exit(1)

# Determine the source file based on hidden activation and softmax activation
if hidden_act == "gelu" and softmax_act == "softmax":
    sys.exit(0)  # No replacement needed
elif hidden_act == "gelu" and softmax_act == "quant":
    source_file = source_dir + "modeling_flax_utils_speedup1.py"
elif hidden_act == "sig" and softmax_act == "softmax":
    source_file = source_dir + "modeling_flax_utils_speedup2.py"
elif hidden_act == "sig" and softmax_act == "quant":
    source_file = source_dir + "modeling_flax_utils_speedup3.py"
else:
    sys.exit(1)  # Invalid combination of arguments

# Backup the target file before replacement
backup_file = target_file + ".bak"
shutil.copyfile(target_file, backup_file)
print(f"The target file has been backed up to {backup_file}。")

# Read the content of the source file
with open(source_file, "r", encoding="utf-8") as f:
    content = f.read()

# Replace the content of the target file with the source file content
with open(target_file, "w", encoding="utf-8") as f:
    f.write(content)

print(f"The content of the file {source_file} has been successfully replaced.")

# Execute the corresponding task script based on the task name
if task_name == "cifar10":
    subprocess.run(["python", "cifar10.py"])
elif task_name == "cifar100":
    subprocess.run(["python", "cifar100.py"])
elif task_name == "tiny-imagenet":
    subprocess.run(["python", "inference_tiny.py"])
elif task_name == "fairface-age":
    subprocess.run(["python", "inference_fair_age.py"])
elif task_name == "fairface-gender":
    subprocess.run(["python", "inference_fair_gender.py"])
elif task_name == "fairface-race":
    subprocess.run(["python", "inference_fair_race.py"])