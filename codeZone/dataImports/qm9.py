# Chandler King
# The purpose of this program is to extract the dataset from the files given on
# https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904
# and turn it into x and y value pairs for machine learning applications

import os
import pickle

from sklearn.model_selection import train_test_split
import pandas as pd

def qm9_data(dataset='train',
             directory='../data/QM9/',
             file_one='dsgdb9nsd.txt',
             output_file=None,
             propertyX='smiles',
             propertyY='free energy at 298.15K'):

    if not (dataset == 'train') | (dataset == 'test'):
        raise ValueError('dataset must equal train or test')

    original_path = directory + '/' + file_one

    # load dataset to pandas dataframe and pickle if file does not exist
    picklefile = os.path.join(directory + '/' + 'qm9' + '_set.pickle')
    if not os.path.isfile(picklefile):
        print("Pickle file doesnt exist. Creating {}".format(picklefile))
        properties = ('smiles', 'tag', 'index', 'rotational constant A',
                      'rotational constant B', 'rotational constant C',
                      'dipole moment', 'isotropic polarizability',
                      'energy of Highest occupied molecular orbital',
                      'energy of Lowest occupied molecular orbital',  # SIC
                      'gap between LUMO and HOMO', 'electronic spatial extent',
                      'zero point vibrational energy', 'internal energy at 0K',
                      'internal energy at 298.15K', 'enthalpy at 298.15K',
                      'free energy at 298.15K', 'heat capacity at 298.15K')
        ovrlist = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                   [], [], []]

        for file in os.listdir(original_path):
            subfiles = original_path + '/' + file
            if not os.path.isfile(subfiles):
                continue
            biz = open(subfiles)
            dat = [line.strip() for line in biz]
            elem = int(dat[0])
            props = dat[1].split()
            ovrlist[0].append(dat[elem + 3])
            for c in range(0, len(props)):
                ovrlist[c + 1].append(props[c])
        d = {}
        for c in range(0, len(ovrlist)):
            d[properties[c]] = ovrlist[c]
            df = pd.DataFrame(data=d, columns=d.keys())
        df.to_pickle(picklefile)

    train_pickle = os.path.join(directory + '/' + 'qm9' + '_train' +
                                '_set.pickle')
    test_pickle = os.path.join(directory + '/' + 'qm9' + '_test' +
                               '_set.pickle')

    numerical_props = ['rotational constant A',
                       'rotational constant B',
                       'rotational constant C',
                       'dipole moment',
                       'isotropic polarizability',
                       'energy of Highest occupied molecular orbital',
                       'energy of Lowest occupied molecular orbital',
                       'gap between LUMO and HOMO',
                       'electronic spatial extent',
                       'zero point vibrational energy',
                       'internal energy at 0K',
                       'internal energy at 298.15K',
                       'enthalpy at 298.15K',
                       'free energy at 298.15K',
                       'heat capacity at 298.15K']

    if not os.path.isfile(train_pickle) or not os.path.isfile(train_pickle):
        with open(picklefile, 'rb') as f:
            df = pickle.load(f)
        print('Loaded pickle file from ' + picklefile)

        if output_file:
            df.to_csv(output_file)

        train_df, test_df = train_test_split(df, test_size = 0.1, random_state = 50)

        train_df = train_df.loc[:, train_df.columns != 'tag']
        train_df = train_df.loc[:,train_df.columns != 'index']

        test_df = test_df.loc[:, test_df.columns != 'tag']
        test_df = test_df.loc[:,test_df.columns != 'index']

        train_smiles = train_df['smiles']
        test_smiles = test_df['smiles']

        train_df = train_df.loc[:, train_df.columns != 'smiles']
        test_df = test_df.loc[:, test_df.columns != 'smiles']

        for c in numerical_props:
            train_df[c] = pd.to_numeric(train_df[c])
            test_df[c] = pd.to_numeric(test_df[c])
        train_df['smiles'] = train_smiles
        test_df['smiles'] = test_smiles

        train_df.to_pickle(train_pickle)
        print("Train pickle file doesnt exist. Created {}".format(train_pickle))

        test_df.to_pickle(test_pickle)
        print("Test pickle file doesnt exist. Created {}".format(test_pickle))

    with open(train_pickle, 'rb') as f:
        train_df = pickle.load(f)
    print('Loaded train pickle file from ' + train_pickle)

    with open(test_pickle, 'rb') as f:
        test_df = pickle.load(f)
    print('Loaded test pickle file from ' + test_pickle)
    # python nx_exp.py --model "1hot_fcnn" --dataset mhc
    # --database "../data/iedb/" --data_filter 'HLA-A*02:01'


    #return keyed values as numpy arrays
    twelve_keys = numerical_props[3:]  # all twelve standard property outputs
    if(dataset == 'train'):
        if propertyY == 'all_12':
            return train_df[propertyX].to_numpy(),\
                train_df[twelve_keys].to_numpy(),\
                twelve_keys
        return train_df[propertyX].to_numpy(), train_df[propertyY].to_numpy()

    if propertyY == 'all_12':
        return test_df[propertyX].to_numpy(),\
            test_df[twelve_keys].to_numpy(),\
            twelve_keys
    return test_df[propertyX].to_numpy(), test_df[propertyY].to_numpy()
