import subprocess
import os
import sys

def run_script(script_name):
    """Runs a python script and prints its output."""
    print(f"--- Running {script_name} ---")
    try:
        # Use sys.executable to ensure the same python interpreter is used
        process = subprocess.Popen([sys.executable, script_name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        process.stdout.close()
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, script_name)
        print(f"--- Finished {script_name} ---")
    except FileNotFoundError:
        print(f"Error: {script_name} not found. Make sure it is in the correct path.")
    except Exception as e:
        print(f"An error occurred while running {script_name}: {e}")


def main():
    """Main function to run the complete pipeline."""
    
    # 1. Generate data if it doesn't exist
    if not os.path.exists('data'):
        run_script('generate_data.py')
    else:
        print("Data directory already exists. Skipping data generation.")

    # 2. Train the model
    run_script('train.py')

    # 3. Evaluate the model
    run_script('evaluate.py')


if __name__ == '__main__':
    main()
