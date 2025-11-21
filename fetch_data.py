import os
import subprocess
import yaml
import sys
import tempfile
import shutil
import time
import argparse
from pathlib import Path

# Remote Host configuration
HOST = "judac"
REMOTE_ROOT_PREFIX = "/p/data1/trustllmd/WP2/data_v2"

# Local directories
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, "data")
MONO_DIR = os.path.join(DATA_DIR, "monolingual")
PARALLEL_DIR = os.path.join(DATA_DIR, "parallel")

# Control Socket for SSH Multiplexing (2FA support)
SSH_CONTROL_DIR = os.path.join(REPO_ROOT, ".ssh_control")
SSH_CONTROL_SOCKET = os.path.join(SSH_CONTROL_DIR, "control_socket")

# Remote Paths (Monolingual)
# Using the specific files provided by the user
REMOTE_FILES = {
    "English": ["/p/data1/trustllmd/WP2/data_v2/tdm/fineweb-edu/CC-MAIN-2022-33/train-00000-of-00017.parquet"],
    "Code": ["/p/data1/trustllmd/WP2/data_v2/code/the-stack-dedup-python/data-00002-of-00144.parquet"],
    "German": ["/p/data1/trustllmd/WP2/data_v2/tdm/fineweb2/deu_Latn/001_00015.parquet"],
    "Swedish": ["/p/data1/trustllmd/WP2/data_v2/tdm/fineweb2/swe_Latn/002_00000.parquet"],
    "Danish": ["/p/data1/trustllmd/WP2/data_v2/tdm/fineweb2/dan_Latn/001_00002.parquet"],
    "Norwegian": ["/p/data1/trustllmd/WP2/data_v2/tdm/fineweb2/nob_Latn/001_00000.parquet"],
    "Dutch": ["/p/data1/trustllmd/WP2/data_v2/tdm/fineweb2/nld_Latn/000_00007.parquet"],
    "Icelandic": [
        "/p/data1/trustllmd/WP2/data_v2/tdm/fineweb2/isl_Latn/000_00000.parquet",
        "/p/data1/trustllmd/WP2/data_v2/hplt2/isl/13_0.parquet"
    ],
    "Faroese": [
        "/p/data1/trustllmd/WP2/data_v2/tdm/fineweb2/fao_Latn/000_00000.parquet",
        "/p/data1/trustllmd/WP2/data_v2/hplt2/fao/1_0.parquet",
        "/p/data1/trustllmd/WP2/data_v2/fao_last_minute/teldni.parquet"
    ],
}

def setup_ssh_master():
    """Establishes a master SSH connection to avoid repeated 2FA prompts."""
    os.makedirs(SSH_CONTROL_DIR, exist_ok=True)
    
    # Clean up old socket if it exists
    if os.path.exists(SSH_CONTROL_SOCKET):
        try:
            os.remove(SSH_CONTROL_SOCKET)
        except OSError:
            pass

    print("--> Establishing Master SSH connection (You will be asked for 2FA ONCE)...")
    print("    Please enter your password/OTP if prompted.")
    
    # -M: Master mode for connection sharing
    # -S: Path to control socket
    # -f: Request ssh to go to background just before command execution
    # -N: Do not execute a remote command (just forward ports/maintain connection)
    cmd = ["ssh", "-M", "-S", SSH_CONTROL_SOCKET, "-f", "-N", HOST]
    
    try:
        subprocess.check_call(cmd)
        print("    Master connection established.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    FAILED to establish master connection: {e}")
        return False

def close_ssh_master():
    """Closes the master SSH connection."""
    print("--> Closing Master SSH connection...")
    cmd = ["ssh", "-S", SSH_CONTROL_SOCKET, "-O", "exit", HOST]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        pass # It might have already closed
    
    if os.path.exists(SSH_CONTROL_DIR):
        shutil.rmtree(SSH_CONTROL_DIR)

def run_ssh_command(cmd_args, description, use_multiplex=True):
    print(f"--> {description}...")
    
    # Insert control socket options if multiplexing is enabled
    final_cmd = list(cmd_args)
    if use_multiplex:
        # For scp, the options come before source/dest
        # For ssh, options come before hostname
        
        # We need to insert '-o ControlPath=...' right after the command name
        # cmd_args[0] is 'ssh' or 'scp'
        
        opts = ["-o", f"ControlPath={SSH_CONTROL_SOCKET}"]
        final_cmd = [final_cmd[0]] + opts + final_cmd[1:]

    try:
        subprocess.check_call(final_cmd)
        print("    Done.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    FAILED: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fetch data from HPC.")
    parser.add_argument("--languages", nargs="+", help="Specific languages to download (e.g. English German). Default: all.")
    args = parser.parse_args()

    # Filter files based on user input
    target_files = REMOTE_FILES
    if args.languages:
        lower_targets = set(l.lower() for l in args.languages)
        target_files = {k: v for k, v in REMOTE_FILES.items() if k.lower() in lower_targets}
        
        if not target_files:
            print(f"No matching languages found for: {args.languages}")
            print(f"Available: {list(REMOTE_FILES.keys())}")
            return

    # 1. Setup SSH Master Connection
    if not setup_ssh_master():
        print("Aborting due to SSH connection failure.")
        return

    try:
        os.makedirs(MONO_DIR, exist_ok=True)
        # os.makedirs(PARALLEL_DIR, exist_ok=True) 

        # Track local paths for config update
        local_data_map = {}

        # 2. Fetch Files
        for lang, paths in target_files.items():
            lang_dir = os.path.join(MONO_DIR, lang.lower())
            os.makedirs(lang_dir, exist_ok=True)
            
            local_paths = []
            for r_path in paths:
                filename = os.path.basename(r_path)
                l_path = os.path.join(lang_dir, filename)
                
                # Check if already exists
                if os.path.exists(l_path):
                    print(f"Skipping {lang} (already exists): {l_path}")
                    local_paths.append(os.path.relpath(l_path, REPO_ROOT))
                    continue

                # SCP
                cmd = ["scp", f"{HOST}:{r_path}", l_path]
                if run_ssh_command(cmd, f"Downloading {lang} data"):
                    local_paths.append(os.path.relpath(l_path, REPO_ROOT))
            
            if local_paths:
                local_data_map[lang] = local_paths

        # 3. Update Config
        config_path = os.path.join(REPO_ROOT, "config", "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update monolingual paths
            if 'monolingual_data' not in config:
                config['monolingual_data'] = {}
                
            for lang, paths in local_data_map.items():
                # Use globs for local paths to match folder content
                dir_path = os.path.dirname(paths[0])
                config['monolingual_data'][lang] = [os.path.join(dir_path, "*.parquet")]

            print("--> Updating config.yaml with local paths...")
            with open(config_path, 'w') as f:
                yaml.dump(config, f, sort_keys=False)
            print("    Done.")

        else:
            print("Warning: config/config.yaml not found.")

    finally:
        # Always close the master connection
        close_ssh_master()

    print(f"\nDownload complete for: {list(target_files.keys())}")

if __name__ == "__main__":
    main()
