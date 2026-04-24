import subprocess
import os
import sys
import argparse

def run_script(script_name, args=None):
    """Runs a python script with arguments and prints its output."""
    print(f"--- Running {script_name} with args: {args} ---")
    try:
        command = [sys.executable, script_name]
        if args:
            command.extend(args)
        # Use sys.executable to ensure the same python interpreter is used
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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
    """Main pipeline to generate data, train the model, and evaluate it."""
    parser = argparse.ArgumentParser(description='Main pipeline for circle detection model.')
    parser.add_argument('--width', type=int, default=128, help='Width of the images for data generation.')
    parser.add_argument('--height', type=int, default=128, help='Height of the images for data generation.')
    parser.add_argument('--noise-level', type=float, default=0.01, help='Noise level for data generation.')
    parser.add_argument('--model', type=str, default='transformer', help='Model to train (transformer or cnn).')
    parser.add_argument('--target_width', type=int, default=128, help='Target width for resizing images.')
    parser.add_argument('--target_height', type=int, default=128, help='Target height for resizing images.')
    args = parser.parse_args()

    # Check if data needs to be generated
    if not os.path.exists('data/train'):
        print("Data not found. Generating dataset...")
        gen_args = [
            '--width', str(args.width),
            '--height', str(args.height),
            '--noise-level', str(args.noise_level)
        ]
        run_script('generate_data.py', gen_args)
    else:
        print("Dataset already exists. Skipping generation.")
    
    train_args = [
        '--model', args.model,
        '--target_width', str(args.target_width),
        '--target_height', str(args.target_height)
    ]
    run_script('train.py', train_args)
    run_script('evaluate.py')

if __name__ == '__main__':
    main()
