from data import *
# import megnet as mg
from megnet.models import MEGNetModel
from megnet.data.crystal import CrystalGraph
import numpy as np
from pymatgen.core.structure import Structure
import os
import csv
from sklearn.model_selection import train_test_split
import tqdm


nfeat_bond = 100
r_cutoff = 5
gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
gaussian_width = 0.5
graph_converter = CrystalGraph(cutoff=r_cutoff)
model = MEGNetModel(graph_converter=graph_converter, centers=gaussian_centers, width=gaussian_width)

# Model training
# Here, `structures` is a list of pymatgen Structure objects.
# `targets` is a corresponding list of properties.
data_path = 'data/'
property='formation_energy'
prop={'formation_energy':1,'band_gap':2,'fermi_energy':3,'total_magnetization':5}
index=prop[property]
radius=8
max_num_nbr = 12
test_size=0.8
full_dataset = CIFData(data_path,max_num_nbr,radius)
datasize=len(full_dataset)
idx_train, idx_test = train_test_split(range(datasize), test_size=test_size, random_state=42)

id_prop_file = os.path.join(data_path, 'id_prop.csv')
with open(id_prop_file) as f:
    reader = csv.reader(f)
    id_prop_data = [row for row in reader]

graphs_valid = []
targets_valid = []
structures_invalid = []
for i in tqdm(idx_train):
    s=Structure.from_file(os.path.join(data_path, str(i) + '.cif'))
    p=float(id_prop_data[i][index])
    try:
        graph = graph_converter.convert(s)
        graphs_valid.append(graph)
        targets_valid.append(p)
    except:
        structures_invalid.append(s)
model.train_from_graphs(graphs_valid, targets_valid,epochs=200)

# Predict the property of a new structure
# graphs_valid_test = []
# true_target_test = []
# for i in idx_test:
#     new_structure=Structure.from_file(os.path.join(data_path,str(i)+'.cif'))
#     try:
#         graph = graph_converter.convert(new_structure)
#         graphs_valid_test.append(graph)
#     except:
#         structures_invalid.append(s)
#     true_target_test.append(float(id_prop_data[i][1]))
# pred_target = model.predict_structure(graphs_valid_test)
ae_list=[]
for i in tqdm(idx_test):
    new_structure=Structure.from_file(os.path.join(data_path,str(i)+'.cif'))
    pred_target = model.predict_structure(new_structure)
    true_target = float(id_prop_data[i][index])
    ae = abs(float(pred_target[0])-true_target)
    ae_list.append(ae)
    print(str(pred_target)+" "+str(true_target)+" "+ str(ae))
print("MAE : "+str(np.mean(ae_list)))
