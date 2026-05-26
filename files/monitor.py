import time
import json
import sys
import os
import kaggle

KERNEL_ID = "goldypahal/bone-fracture-training"
LOG_FILE = "training_logs.txt"

def get_status_and_logs():
    api = kaggle.KaggleApi()
    api.authenticate()
    
    # Check kernel status
    status_obj = api.kernels_status(KERNEL_ID)
    status = str(status_obj.status).split('.')[-1].upper()
    failure_msg = getattr(status_obj, "failure_message", "") or ""
    
    # Check logs
    logs_raw = api.kernels_logs(KERNEL_ID)
    log_lines = []
    if logs_raw.strip():
        try:
            data = json.loads(logs_raw)
            log_lines = [x.get("data", "") for x in data]
        except Exception:
            pass
            
    return status, failure_msg, log_lines

def main():
    print("=" * 60)
    print(f"  Kaggle Kernel Monitoring Script Started")
    print(f"  Target: {KERNEL_ID}")
    print(f"  Interval: 60 seconds")
    print(f"  Saving full logs locally to: {LOG_FILE}")
    print("=" * 60)

    # Initialize log file immediately to prevent file-not-found errors
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("[Initializing Kaggle container and mounting datasets... Please wait a few minutes.]\n")

    last_log_count = 0
    consecutive_errors = 0

    while True:
        try:
            status, failure_msg, log_lines = get_status_and_logs()
            consecutive_errors = 0
            
            # Print current status
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{current_time}] Status: {status}")
            
            if failure_msg:
                print(f"  ⚠️ Failure Message: {failure_msg}")

            # Stream new logs
            if len(log_lines) > last_log_count:
                new_lines = log_lines[last_log_count:]
                new_content = "".join(new_lines)
                sys.stdout.buffer.write(new_content.encode("utf-8"))
                sys.stdout.flush()
                last_log_count = len(log_lines)
                
                # Update local log file
                with open(LOG_FILE, "w", encoding="utf-8") as f:
                    f.write("".join(log_lines))

            # Stop condition
            if status not in ["RUNNING", "QUEUED", "UNKNOWN"]:
                print(f"\n[!] Execution finished with final status: {status}")
                if failure_msg:
                    print(f"[!] Error: {failure_msg}")
                break
                
        except Exception as e:
            consecutive_errors += 1
            print(f"[!] Error during check: {e}")
            if consecutive_errors > 10:
                print("[!] Too many consecutive errors. Exiting.")
                break
                
        # Sleep for 60 seconds
        time.sleep(60)

if __name__ == "__main__":
    main()
