import numpy as np
import pandas as pd
import os

tces=pd.DataFrame.from_csv('TCE_dets_per_sector_fixed_dur_and_labels.csv')

nglob=1001
nloc=101

data_dir='/home/hosborn/TESS/processed_dvpdc_both'
if not os.path.isdir(data_dir):
    os.system('mkdir '+data_dir)
shuffled_ints=np.arange(len(tces))
np.random.shuffle(shuffled_ints)
for n,i in enumerate(np.array_split(shuffled_ints,10)):
    if n==9:
        labels='test'
    else:
        labels='all'
    if not os.path.isdir(os.path.join(data_dir,labels)):
        os.system('mkdir '+os.path.join(data_dir,labels))

    csvname='/home/hosborn/TESS/csvs/tce_'+str(n)+'_'+labels+'.csv'
    tces.iloc[i].to_csv(csvname)
    with open("/home/hosborn/TESS/run_preprocess"+str(n)+".sh",'w') as f:
        f.write("source /home/hosborn/TESS2/bin/activate \n")
        f.write("export PATH=$PATH:~/home/hosborn/TESS/astronet \n")
        f.write("cd /home/hosborn/TESS \n")
        f.write("python QuickProcess_new.py "+csvname+" "+labels+" "+data_dir+" "+str(int(nglob))+" "+str(int(nloc)))
    if n==0:
        with open('/home/hosborn/TESS/run_all_preprocesses.sh','w') as f:
            f.write("qsub -l nodes=1,walltime=24:00:00,mem=8GB /home/hosborn/TESS/run_preprocess"+str(n)+".sh \n")
    else:
        with open('/home/hosborn/TESS/run_all_preprocesses.sh','a') as f:
            f.write("qsub -l nodes=1,walltime=24:00:00,mem=8GB /home/hosborn/TESS/run_preprocess"+str(n)+".sh \n")

