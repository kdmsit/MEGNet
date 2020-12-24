from data import *
# import megnet as mg
from megnet.models import MEGNetModel
from megnet.data.crystal import CrystalGraph
import numpy as np
from pymatgen.core.structure import Structure
import os
import csv
from sklearn.model_selection import train_test_split


nfeat_bond = 100
r_cutoff = 5
gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
gaussian_width = 0.5
graph_converter = CrystalGraph(cutoff=r_cutoff)
# graph_converter = full_dataset(cutoff=r_cutoff)
model = MEGNetModel(graph_converter=graph_converter, centers=gaussian_centers, width=gaussian_width)
# model = MEGNetModel(centers=gaussian_centers, width=gaussian_width)

# Model training
# Here, `structures` is a list of pymatgen Structure objects.
# `targets` is a corresponding list of properties.
data_path = 'data_100/'
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

structures = []
targets = []
for i in idx_train:
    structures.append(Structure.from_file(os.path.join(data_path,str(i)+'.cif')))
    targets.append(id_prop_data[i][1])
print(targets)
model.train(structures, targets, epochs=1000)

# Predict the property of a new structure
for i in idx_test:
    new_structure=Structure.from_file(os.path.join(data_path,str(i)+'.cif'))
    pred_target = model.predict_structure(new_structure)
    true_target = id_prop_data[i][1]
    print(pred_target,true_target)
