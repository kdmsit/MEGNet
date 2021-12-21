from matminer.utils.io import load_dataframe_from_json
import pickle as pkl
import os

mp_e_form = load_dataframe_from_json("../../npj_data/matbench_mp_e_form.json.gz")    #matbench_mp_e_form
# mp_e_form = load_dataframe_from_json("../matbench_jdft2d.json.gz")    #matbench_mp_e_form
# print(mp_e_form.shape[0])
directory = "../new_data/"
k=0
for index, row in mp_e_form.iterrows():
    # print(row)
    # exit()
    struct=row.structure
    property=row.e_form
    path = directory + str(k)
    if not os.path.exists(path):
        pkl_out = open(path, "wb")
        pkl.dump([struct,
                  property], pkl_out)
        pkl_out.close()
        k = k + 1
print(k)