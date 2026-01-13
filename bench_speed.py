import time
import subprocess
import sys

def run_bench():
    start = time.time()
    # Using grad_accum 8 and batch_size 4
    # We'll run for 50 steps
    cmd = [sys.executable, "-u", "scripts/train_super_math.py", "--steps", "50", "--grad_accum", "8"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    steps_started = 0
    first_step_time = None
    
    for line in process.stdout:
        print(line, end='')
        if "Step" in line:
            if first_step_time is None:
                first_step_time = time.time()
            steps_started += 1
            if steps_started >= 50:
                break
    
    process.terminate()
    end = time.time()
    
    if first_step_time:
        duration = end - first_step_time
        steps = steps_started - 1 # Since we start timing AFTER the first step printing started (approx)
        if steps > 0:
            sps = steps / duration
            print(f"\nBenchmark Results:")
            print(f"Steps per second: {sps:.4f}")
            print(f"Seconds per step: {1/sps:.4f}")
            print(f"Steps per hour: {sps * 3600:.2f}")

if __name__ == "__main__":
    run_bench()
