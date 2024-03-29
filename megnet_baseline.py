from data import *
# import megnet as mg
from megnet.models import MEGNetModel
from megnet.data.crystal import CrystalGraph
import numpy as np
from pymatgen.core.structure import Structure
import os
import csv
import pickle as pkl
import pandas as pd
from sklearn.model_selection import train_test_split
import tqdm


nfeat_bond = 100
epoch=200
r_cutoff = 5
gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
gaussian_width = 0.5
graph_converter = CrystalGraph(cutoff=r_cutoff)
model = MEGNetModel(graph_converter=graph_converter, centers=gaussian_centers, width=gaussian_width)
print(model.summary())


# Model training
# Here, `structures` is a list of pymatgen Structure objects.
# `targets` is a corresponding list of properties.
data_path = '../new_data/'

# property='bm'
# prop={'bm':1,'sm':2,'pr':3}
property='formation_energy'
prop={'formation_energy':1,'band_gap':2,'fermi_energy':3,'total_magnetization':5}
index=prop[property]
radius=8
max_num_nbr = 12
test_size=0.4
full_dataset = CIFData(data_path,max_num_nbr,radius)
datasize=636 #132752
idx_train, idx_test = train_test_split(range(datasize), test_size=test_size, random_state=42)
idx_val, idx_test = train_test_split(idx_test, test_size=0.5, random_state=42)
print("Test size " + str(len(idx_test)))
print("Train size " + str(len(idx_train)))

# id_prop_file = os.path.join(data_path, 'id_prop.csv')
# with open(id_prop_file) as f:
#     reader = csv.reader(f)
#     id_prop_data = [row for row in reader]
#
# hash_file = os.path.join(data_path, 'material_id_hash.csv')
# with open(hash_file) as f:
#     reader = csv.reader(f)
#     cif_mpid = [row for row in reader]
#
# for i in idx_test:
#     mpid=cif_mpid[i][1]

print("Train Data Load Started......")
graphs_valid = []
targets_valid = []
structures_invalid = []
for i in idx_train:
    # s=Structure.from_file(os.path.join(data_path, str(i) + '.cif'))
    # p=float(id_prop_data[i][index])
    file_path = os.path.join(data_path, str(i))
    with open(file_path, 'rb') as handle:
        sentences = pkl.load(handle)
    s, p = sentences[0], sentences[1]
    # targets = []
    # targets.append(property)
    # p = np.log10(float(id_prop_data[i][index]))
    try:
        graph = graph_converter.convert(s)
        graphs_valid.append(graph)
        targets_valid.append(p)
    except:
        structures_invalid.append(s)
print("Train Data Load Done......")

print("Training the model......")
model.train_from_graphs(graphs_valid, targets_valid,epochs=epoch)

print("Generate Testing Results......")

final_test_list=[["Cif","True","Predicted","Absolute Error"]]
ae_list=[]
for i in idx_test:
    try:
        new_structure = Structure.from_file(os.path.join(data_path, str(i) + '.cif'))
        pred_target = model.predict_structure(new_structure)
        true_target = float(id_prop_data[i][index])
        ae = abs(float(pred_target[0])-true_target)
        ae_list.append(ae)
        # print(str(pred_target)+" "+str(true_target)+" "+ str(ae))
        final_test_list.append([i,pred_target[0], true_target, ae])
    except:
        # structures_invalid.append(new_structure)
        continue
print("MAE : "+str(np.mean(ae_list)))
# my_df = pd.DataFrame(final_test_list)
# my_df.to_csv('test_'+property+'.csv', index=False, header=False)



