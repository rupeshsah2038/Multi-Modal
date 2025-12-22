#!/usr/bin/env python3
"""
Ensure each medpix config has a wound counterpart and vice-versa in the target folder.

Behavior:
- Removes any leftover '*.tmp.yaml' files in the folder.
- For any config named 'medpix-<suffix>.yaml' without 'wound-<suffix>.yaml', creates a wound copy
  with the `data` block adapted for wound (root, labels, columns).
- For any config named 'wound-<suffix>.yaml' without 'medpix-<suffix>.yaml', creates a medpix copy
  with the `data` block adapted for medpix.

This makes counts symmetric; review generated files before running experiments.
"""
import glob, os, yaml, shutil

DIR = 'config/ultra-edge-hp-tuned-all'

def normalize_data_for_wound(doc):
    doc['data'] = {
        'type': 'wound',
        'root': 'datasets/Wound-1-0',
        'batch_size': 16,
        'num_workers': 4,
        'task1_label': 'type',
        'task2_label': 'severity',
        'type_column': 'type',
        'severity_column': 'severity',
        'description_column': 'description',
        'filepath_column': 'img_path'
    }
    return doc

def normalize_data_for_medpix(doc):
    doc['data'] = {
        'type': 'medpix',
        'root': 'datasets/MedPix-2-0',
        'batch_size': 16,
        'num_workers': 4,
        'task1_label': 'modality',
        'task2_label': 'location'
    }
    return doc

def main():
    files = sorted(glob.glob(os.path.join(DIR, '*.yaml')))
    # remove tmp files
    for f in files:
        if f.endswith('.tmp.yaml'):
            print('Removing temp file', f)
            os.remove(f)

    files = sorted(glob.glob(os.path.join(DIR, '*.yaml')))
    med_suffixes = {}
    wound_suffixes = {}
    for f in files:
        name = os.path.basename(f)
        if name.startswith('medpix-'):
            suffix = name.split('-',1)[1]
            med_suffixes[suffix] = f
        elif name.startswith('wound-'):
            suffix = name.split('-',1)[1]
            wound_suffixes[suffix] = f

    # create missing wound from medpix
    for suffix, medf in med_suffixes.items():
        if suffix not in wound_suffixes:
            print('Creating wound counterpart for', medf)
            with open(medf,'r') as rf:
                doc = yaml.safe_load(rf)
            doc = normalize_data_for_wound(doc)
            out = os.path.join(DIR, f'wound-{suffix}')
            with open(out,'w') as wf:
                yaml.safe_dump(doc, wf, sort_keys=False)

    # create missing medpix from wound
    for suffix, woundf in wound_suffixes.items():
        if suffix not in med_suffixes:
            print('Creating medpix counterpart for', woundf)
            with open(woundf,'r') as rf:
                doc = yaml.safe_load(rf)
            doc = normalize_data_for_medpix(doc)
            out = os.path.join(DIR, f'medpix-{suffix}')
            with open(out,'w') as wf:
                yaml.safe_dump(doc, wf, sort_keys=False)

    # report final counts
    files = sorted(glob.glob(os.path.join(DIR, '*.yaml')))
    med = [f for f in files if os.path.basename(f).startswith('medpix-')]
    wound = [f for f in files if os.path.basename(f).startswith('wound-')]
    print('Final counts -> medpix:', len(med), 'wound:', len(wound))

if __name__ == '__main__':
    main()
