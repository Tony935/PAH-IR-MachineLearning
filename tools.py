# -*- coding: utf-8 -*-
"""
:File: tools.py
:Author: 周东来
:Email: zhoudl@mail.ustc.edu.cn
"""
import os
from openbabel import pybel

import numpy as np
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprint


def spectrum_processing(spec):
    spec = np.delete(spec, spec[:, 1] == 0, axis=0)
    drop = []
    for a in range(spec.shape[0] - 1):
        if spec[a, 0] == spec[a + 1, 0]:
            if spec[a, 1] > spec[a + 1, 1]:
                drop.append(a + 1)
            else:
                drop.append(a)
    spec = np.delete(spec, drop, axis=0)
    return spec


def preprocessing(pahdb):
    with open(pahdb, 'r', encoding="UTF-8") as file:
        lines = file.readlines()

    db = []
    loc = 0
    uid, form, stru, spec = None, None, None, None
    for i in range(len(lines)):
        if 'UID' in lines[i]:
            uid = int(lines[i].split()[1])

        if 'FORMULA' in lines[i]:
            form = lines[i].split()[1]

        if 'GEOMETRY' in lines[i]:
            loc = i

        if 'TRANSITIONS' in lines[i]:
            stru = np.array([lines[j].split() for j in range(loc + 2, i)], dtype='float64')
            loc = i

        if '#' * 79 in lines[i] and loc > 0 and i - loc > 1:
            spec = np.array([lines[j].split()[:2] for j in range(loc + 3, i)], dtype='float64')
            spec = spectrum_processing(spec)
            db.append({'uid': uid, 'formula': form, 'structure': stru, 'spectrum': spec})
            loc = i

    return np.array(db)


def get_xyz_file(db, save_dir):
    atom = {
        1: ' H',
        6: ' C',
        7: ' N',
        8: ' O',
        12: 'Mg',
        14: 'Si',
        26: 'Fe',
    }

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i in db:
        xyz = [f"{i['structure'].shape[0]}\n", f"{i['formula']}\n"]
        for a in i['structure']:
            xyz.append(f"{atom[a[-1]]}{a[1]:12.6f}{a[2]:12.6f}{a[3]:12.6f}\n")
        with open(f"{save_dir}/{i['uid']}.xyz", 'w') as file:
            file.write(''.join(xyz))


def get_mol_file(xyz_dir, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for file in os.listdir(xyz_dir):
        mol = next(pybel.readfile('xyz', f"{xyz_dir}/{file}"))
        mol.write('mol', f"{save_dir}/{file.split('.')[0]}.mol")


def get_fingerprint(mol_dir, radius=4):
    fp = {}
    for file in os.listdir(mol_dir):
        uid = int(file.split('.')[0])
        mol = Chem.MolFromMolFile(f"{mol_dir}/{file}")
        bi = {}
        GetMorganFingerprint(mol, radius=radius, bitInfo=bi, useFeatures=False, useBondTypes=False)
        fp[uid] = np.array(list(bi.keys()))
    return fp
