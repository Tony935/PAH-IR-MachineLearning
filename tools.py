# -*- coding: utf-8 -*-
"""
:File: tools.py
:Author: 周东来
:Email: zhoudl@mail.ustc.edu.cn
"""
import os
from openbabel import pybel

import numpy as np
import pandas as pd
import tensorflow as tf
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


def get_xyz_file(pahdb, save_dir):
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

    for i in pahdb:
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


def get_fingerprint(mol_dir, radius):
    fp = {}
    for file in os.listdir(mol_dir):
        uid = int(file.split('.')[0])
        mol = Chem.MolFromMolFile(f"{mol_dir}/{file}")
        ecfp = GetMorganFingerprint(mol, radius=radius, useFeatures=False, useBondTypes=False).GetNonzeroElements()
        fp[uid] = np.array([list(ecfp.keys()), list(ecfp.values())]).T
    return fp


def banning(pahdb, ban_el):
    elements = None
    if ban_el == 0:
        elements = []
    elif ban_el == 1:
        elements = ['N', 'O', 'Mg', 'Si', 'Fe']
    elif ban_el == 2:
        elements = ['+', '-', 'N', 'O', 'Mg', 'Si', 'Fe']

    bans = []
    for i in range(pahdb.shape[0]):
        for e in elements:
            if e in pahdb[i]['formula']:
                bans.append(i)
                break

    return np.delete(pahdb, bans)


def get_onehot_fingerprint(db_train, db_test):
    train_size = db_train.shape[0]
    test_size = db_test.shape[0]

    max_len = max(i['fingerprint'].shape[0] for i in db_train)
    fp_set = np.delete(np.unique(
        [np.pad(i['fingerprint'][:, 0], [0, max_len - i['fingerprint'].shape[0]]) for i in db_train]), 0)

    fp_train = pd.DataFrame(np.zeros((train_size, fp_set.shape[0]), dtype=np.int32), columns=fp_set)
    fp_test = pd.DataFrame(np.zeros((test_size, fp_set.shape[0]), dtype=np.int32), columns=fp_set)

    for i in range(train_size):
        for fp in db_train[i]['fingerprint'][:, 0]:
            fp_train[fp][i] = 1
    for i in range(test_size):
        for fp in db_test[i]['fingerprint'][:, 0]:
            if fp in fp_set:
                fp_test[fp][i] = 1

    return tf.sparse.from_dense(fp_train), tf.sparse.from_dense(fp_test)


def get_binned_spectrum(db_train, db_test, bins):
    train_size = db_train.shape[0]
    test_size = db_test.shape[0]

    max_fre_len = max(i['spectrum'].shape[0] for i in db_train)
    fre = pd.DataFrame(np.array(
        [np.pad(i['spectrum'][:, 0], [0, max_fre_len - i['spectrum'].shape[0]], constant_values=np.nan) for i in
         db_train]).ravel())
    fre['cut'] = pd.cut(fre[0], bins, precision=5)
    cut = fre['cut'].value_counts(sort=False)

    spec_train = pd.DataFrame(np.zeros((train_size, bins)), columns=cut.index)
    for i in range(train_size):
        isum = db_train[i]['spectrum'][:, 1].sum()
        for j in range(db_train[i]['spectrum'].shape[0]):
            spec_train[fre['cut'][i * max_fre_len + j]][i] += db_train[i]['spectrum'][j, 1] / isum

    rights = [i.right for i in cut.index]
    spec_test = pd.DataFrame(np.zeros((test_size, bins)), columns=cut.index)
    for i in range(test_size):
        isum = db_test[i]['spectrum'][:, 1].sum()
        col = 0
        for j in range(db_test[i]['spectrum'].shape[0]):
            while db_test[i]['spectrum'][j, 0] > rights[col] and col < bins - 1:
                col += 1
            spec_test.iloc[i, col] += db_test[i]['spectrum'][j, 1] / isum

    return spec_train, spec_test
