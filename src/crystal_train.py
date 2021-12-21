import numpy as np
from monty.json import MontyDecoder
from data import *
from tqdm import tqdm
from monty.serialization import loadfn
import pickle as pkl
import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from megnet.data.crystal import CrystalGraph
from megnet.data.graph import GaussianDistance
from megnet.models import MEGNetModel
import argparse

# data = loadfn('bulk_moduli.json')
# structures = data['structures']
# targets = np.log10(data['bulk_moduli'])
parser = argparse.ArgumentParser()
parser.add_argument('--data-path', type=str, default='../../new_data/', help='Root Data Path')
parser.add_argument('--property', type=str, default='formation_energy', help='Property')
parser.add_argument('--test-ratio', type=float, default=0.4, help='Test Split')
parser.add_argument('--epoch', type=int, default=200, help='Number of Training Epoch')
args = parser.parse_args()

data_path = args.data_path
property=args.property
prop={'formation_energy':1,'band_gap':2,'fermi_energy':3,'total_magnetization':5}
# prop={'bm':1,'sm':2,'pr':3}
index=prop[property]
radius=8
max_num_nbr = 12
test_size=args.test_ratio
full_dataset = CIFData(data_path,max_num_nbr,radius)
datasize=132752 #636
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

graph_converter=CrystalGraph(bond_converter=GaussianDistance(np.linspace(0, 8, 10), 0.2))
structures=[]
targets=[]
for i in tqdm(range(len(idx_train))):
        file_path = os.path.join(data_path, str(idx_train[i]))
        with open(file_path, 'rb') as handle:
            sentences = pkl.load(handle)
        s, p = sentences[0], sentences[1]
        try:
            graph = graph_converter.convert(s)
            structures.append((s))
            # p = float(id_prop_data[i][index])
            # p=np.log10(float(id_prop_data[i][index]))
            targets.append(p)
        except:
            continue


model = MEGNetModel(10, 2, nblocks=1, lr=1e-2,
                    n1=4, n2=4, n3=4, npass=1, ntarget=1,
                    graph_converter=CrystalGraph(bond_converter=GaussianDistance(np.linspace(0, 5, 10), 0.5)))

model.train(structures, targets, epochs=args.epoch)

# Test
final_test_list=[["Cif","Mpid","True","Predicted","Absolute Error"]]
ae_list=[]
for i in tqdm(range(len(idx_test))):
    try:
        # mpid = cif_mpid[i][1]
        # new_structure = Structure.from_file(os.path.join(data_path, str(i) + '.cif'))

        file_path = os.path.join(data_path, str(idx_train[i]))
        with open(file_path, 'rb') as handle:
            sentences = pkl.load(handle)
        s, true_target = sentences[0], sentences[1]


        pred_target = model.predict_structure(s).ravel()
        # true_target = float(id_prop_data[i][index])
        # true_target = np.log10(float(id_prop_data[i][index]))
        ae = abs(float(pred_target[0])-true_target)
        ae_list.append(ae)
        # final_test_list.append([i, mpid,pred_target[0], true_target, ae])
        # print(str(mpid)+str(pred_target[0])+" "+str(true_target)+" "+ str(ae))
    except:
        # structures_invalid.append(new_structure)
        continue
print("MAE : "+str(np.mean(ae_list)))
# my_df = pd.DataFrame(final_test_list)
# my_df.to_csv('test_'+property+'.csv', index=False, header=False)


# import pickle as pkl
# pkl_out=open("model_bg","wb")
# pkl.dump(model,pkl_out)
# pkl_out.close()
# torch.save(model, "model_bg")