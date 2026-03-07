import subprocess
import random
for _ in range(100):
    i = random.randint(1, 200)
    subprocess.run(["python", "crawler_logged.py", "--chunk=" + str(i), "--total-chunks=200"])