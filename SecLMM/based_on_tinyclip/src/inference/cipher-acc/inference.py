import shutil
import sys
import subprocess

source_dir = "SecLMM/based_on_tinyclip/src/training/model/"
target_file = "/home/na/miniconda3/envs/seclmm/lib/python3.9/site-packages/transformers/models/clip/modeling_flax_clip.py"

args = sys.argv[1:]  
task_name = None
hidden_act = None
softmax_act = None


for i in range(len(args)):
    if args[i] == "--task_name":
        task_name = args[i + 1]
    elif args[i] == "--hidden_act":
        hidden_act = args[i + 1]
    elif args[i] == "--softmax_act":
        softmax_act = args[i + 1]


valid_task_names = ["cifar10", "cifar100", "tiny-imagenet", "fairface-age", "fairface-gender", "fairface-race"]
if task_name not in valid_task_names:
    print(f"Invalid tasks: {', '.join(valid_task_names)}。")
    sys.exit(1)

if hidden_act == "gelu" and softmax_act == "softmax":
    sys.exit(0)
elif hidden_act == "gelu" and softmax_act == "quant":
    source_file = source_dir + "modeling_flax_utils_speedup1.py"
elif hidden_act == "sig" and softmax_act == "softmax":
    source_file = source_dir + "modeling_flax_utils_speedup2.py"
elif hidden_act == "sig" and softmax_act == "quant":
    source_file = source_dir + "modeling_flax_utils_speedup3.py"
else:
    sys.exit(1)

backup_file = target_file + ".bak"
shutil.copyfile(target_file, backup_file)
print(f"The target file has been backed up to {backup_file}。")


with open(source_file, "r", encoding="utf-8") as f:
    content = f.read()


with open(target_file, "w", encoding="utf-8") as f:
    f.write(content)

print(f"The content of the file {source_file} has been successfully replaced.")

if task_name == "cifar10":
    subprocess.run(["python", "acc_cipher_cifar10.py"])
elif task_name == "cifar100":
    subprocess.run(["python", "acc_cipher_cifar100.py"])
elif task_name == "tiny-imagenet":
    subprocess.run(["python", "acc_cipher_tinyimage.py"])
elif task_name == "fairface-age":
    subprocess.run(["python", "acc_cipher_fair_age.py"])
elif task_name == "fairface-gender":
    subprocess.run(["python", "acc_cipher_fair_gender.py"])
elif task_name == "fairface-race":
    subprocess.run(["python", "acc_cipher_fair_race.py"])
