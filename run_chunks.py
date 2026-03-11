total_runs = 100
import subprocess
import random
for _ in range(total_runs):
    i = random.randint(1, total_runs)
    subprocess.run(["python", "crawler_logged.py", "--chunk=" + str(i), "--total-chunks=" + str(total_runs)])