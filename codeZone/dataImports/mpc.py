import os
import sys
import pickle
import pymatgen as mg
from pymatgen import MPRester

from sklearn.model_selection import train_test_split
import pandas as pd


def mpc_data(dataset='train', directory='../data/mpc/', output_file=None,
             properties=None):

    # single big vector option with y being a big vector with no labels
    # x being cif - graph derived on cif information --- This was a downstream task
    # solve it in mpc for graph type return
    # Solve after [terrace model]
    
    supported_task_properties = ("energy", "energy_per_atom", "volume",
                                 "formation_energy_per_atom", "nsites",
                                 "unit_cell_formula", "pretty_formula",
                                 "is_hubbard",
                                 "elements", "nelements", "e_above_hull",
                                 "hubbards",
                                 "is_compatible", "spacegroup",
                                 "band_gap", "density", "icsd_id", "cif")
    # atom coords not made yet
    additional_properties = ('cell_length', 'cell_angle', 'atom_site', 'atom_coords')

    if not (dataset == 'train') | (dataset == 'test'):
        raise ValueError('dataset must equal train or test')
    #if (propertyX not in supported_task_properties and propertyX != ''):
    #   raise ValueError("propertyX must be in supported_task_properties or empty")
    #if (propertyY not in supported_task_properties and propertyY != ''):
    #    raise ValueError("propertyY must be in supported_task_properties or empty")

    unorganized_file = os.path.join(directory + '/' + 'mpc_unorganized.pickle')
    if not os.path.isfile(unorganized_file):
        elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
                    'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
                    'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
                    'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
                    'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
                    'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
                    'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                    'Pb', 'Bi', 'Po', 'At', 'Rn', 'Ac', 'Th', 'Pa', 'U', 'Np',
                    'Pu']

        # need generated API KEY
        # from https://materialsproject.org/janrain/loginpage/?next=/dashboard
        API_KEY = "QPlTweIuFN0oTR4I7xB"
        # object for going through the Materials Project
        mpr = MPRester(API_KEY)
        entries = mpr.query({
            "elements": {
                '$in': elements
            }
        }, supported_task_properties)
        print("Pickel file doesnt exist. Creating {}".format(unorganized_file))
        file = open(unorganized_file, 'wb')
        pickle.dump(entries, file)
        file.close()

    picklefile = os.path.join(directory + '/' + 'mpc' + '_set.pickle')
    if not os.path.isfile(picklefile):
        print("Pickle file doesnt exist. Creating {}".format(picklefile))
        with open(unorganized_file, 'rb') as f:
            mpc = pickle.load(f)
        this_dict = {}
        this_list = [[], [], [], [], [], [], [], [], [], [], [], [], [], [],
                     [], [], [], []]

        for dict in mpc:
            index = 0
            for key, value in dict.items():
                this_list[index].append(value)
                index = index + 1
                # df = df.append({key : value}, ignore_index=True)
        for c in range(0, len(supported_task_properties)):
            this_dict[supported_task_properties[c]] = this_list[c]
        df = pd.DataFrame(data=this_dict, columns=this_dict.keys())
        df.to_pickle(picklefile)

    print('Loaded pickle file from ' + picklefile)
    with open(picklefile, 'rb') as f:
        df = pickle.load(f)

    if output_file:
        df.to_csv(output_file)

    train_pickle = os.path.join(directory + '/' + 'mpc' + '_train' +
                                '_set.pickle')
    test_pickle = os.path.join(directory + '/' + 'mpc' + '_test' +
                               '_set.pickle')
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=50)

    if not os.path.isfile(train_pickle):
        print("Train pickle file doesnt exist. Creating {}"
              .format(train_pickle))
        train_df.to_pickle(train_pickle)
    if not os.path.isfile(test_pickle):
        print("Test pickle file doesnt exist. Creating {}"
              .format(test_pickle))
        test_df.to_pickle(test_pickle)

    print('Loading train pickle file from ' + train_pickle)
    with open(train_pickle, 'rb') as f:
        train_df = pickle.load(f)

    print('Loading test pickle file from ' + test_pickle)
    with open(test_pickle, 'rb') as f:
        test_df = pickle.load(f)

    train_cif_break = train_df['cif'].to_numpy()
    test_cif_break = test_df['cif'].to_numpy()

    for c in range(0,1):
        string = train_cif_break[c]
        splitz = string.splitlines()
        cell_a = splitz[3].split(' ', 1)[1]
        cell_b = splitz[4].split(' ', 1)[1]
        cell_c = splitz[5].split(' ', 1)[1]
        train_df[additional_properties[0]] = cell_a + ', ' + cell_b + ', ' + cell_c
        ca = splitz[6].split(' ', 1)[1]
        cb = splitz[7].split(' ', 1)[1]
        cc = splitz[8].split(' ', 1)[1]
        train_df[additional_properties[1]] = ca + ', ' + cb + ', ' + cc
        lis = []
        for a in range(26, len(splitz)):
            lis.append(splitz[a])
        print(lis)
        train_df[additional_properties[2]] = pd.Series(lis)


    return train_df[properties]
    # x value will always be cif
    #



    pymatgen_cite = '''
        Shyue Ping Ong, William Davidson Richards, Anubhav Jain,
        Geoffroy Hautier, Michael Kocher, Shreyas Cholia, Dan Gunter,
        Vincent Chevrier, Kristin A. Persson, Gerbrand Ceder.
        Python Materials Genomics (pymatgen) : A Robust, Open-Source
        Python Library for Materials Analysis.
        Computational Materials Science, 2013, 68, 314-319.
        doi:10.1016/j.commatsci.2012.10.028
        '''
