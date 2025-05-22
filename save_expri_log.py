import argparse
import os
from datetime import datetime

parser = argparse.ArgumentParser(description="Save experiment log with timestamp")
parser.add_argument("--messages", type=str, required=True, help="Message to log")
args = parser.parse_args()

log_dir = "./scripts"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "expri.log")

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_entry = f"[{timestamp}] {args.messages}\n"

with open(log_file, "a", encoding="utf-8") as f:
    f.write(log_entry)

print(f"Logged: {log_entry.strip()}")
