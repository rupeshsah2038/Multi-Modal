#!/usr/bin/env python3
"""
Rerun configs listed in failed_runs.txt

Usage:
  python tools/rerun_failed_runs.py --gpu cuda:0  # run on GPU
  python tools/rerun_failed_runs.py --cpu --dry   # print commands and set device to cpu
"""
import os
import subprocess
import sys
import argparse


def read_failed(file_path='failed_runs.txt'):
    if not os.path.exists(file_path):
        print(f'{file_path} not found')
        return []
    cfgs = []
    with open(file_path,'r') as f:
        for line in f:
            line=line.strip()
            if not line: 
                continue
            parts=line.split('\t',1)
            cfgs.append(parts[0])
    return cfgs


def override_device_and_run(cfg_path, device, dry=False):
    with open(cfg_path,'r') as f:
        lines = f.readlines()
    new_lines = []
    found=False
    for line in lines:
        if line.strip().startswith('device:'):
            new_lines.append(f'device: "{device}"\n')
            found=True
        else:
            new_lines.append(line)
    if not found:
        new_lines.append(f'\ndevice: "{device}"\n')
    tmp = cfg_path + '.tmp.yaml'
    with open(tmp,'w') as f:
        f.writelines(new_lines)
    cmd = [sys.executable, 'experiments/run.py', tmp]
    print('Running:', ' '.join(cmd))
    if not dry:
        subprocess.run(cmd)
    os.remove(tmp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, help='Device string, e.g. cuda:0')
    parser.add_argument('--cpu', action='store_true', help='Run on cpu')
    parser.add_argument('--dry', action='store_true', help='Print commands but do not execute')
    parser.add_argument('--file', type=str, default='failed_runs.txt')
    args = parser.parse_args()

    cfgs = read_failed(args.file)
    if not cfgs:
        print('No failed configs to run.')
        return
    device = 'cpu' if args.cpu else (args.gpu or 'cuda:0')
    for cfg in cfgs:
        override_device_and_run(cfg, device, dry=args.dry)


if __name__ == '__main__':
    main()
