from megnet.utils.models import load_model
from pymatgen.core.structure import Structure
from sklearn.model_selection import train_test_split
from data import *

model = load_model("Eform_MP_2019")

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

hash_file = os.path.join(data_path, 'material_id_hash.csv')
with open(hash_file) as f:
    reader = csv.reader(f)
    cif_mpid = [row for row in reader]
final_test_list=[["Cif","Mpid","True","Predicted","Absolute Error"]]
ae_list=[]
for i in idx_test:
    try:
        mpid = cif_mpid[i][1]
        new_structure = Structure.from_file(os.path.join(data_path, str(i) + '.cif'))
        pred_target = model.predict_structure(new_structure).ravel()
        true_target = float(id_prop_data[i][index])
        ae = abs(float(pred_target[0])-true_target)
        ae_list.append(ae)
        final_test_list.append([i, mpid,pred_target[0], true_target, ae])
        print(str(mpid)+str(pred_target)+" "+str(true_target)+" "+ str(ae))
    except:
        # structures_invalid.append(new_structure)
        continue
print("MAE : "+str(np.mean(ae_list)))
my_df = pd.DataFrame(final_test_list)
my_df.to_csv('test_'+property+'.csv', index=False, header=False)



