import numpy as np
import os
print(os.path.abspath(os.path.dirname(__file__)))
deepmdsr_sA = np.loadtxt('/root/app/deepmd-les/deepmd-les/DeePMD-kit-FastLearn/campaign_water_interface/deepmd/runs/deepmd_sA/s7/lcurve.out', usecols=(0,3,5))
deepmdsr_sB = np.loadtxt('/root/app/deepmd-les/deepmd-les/DeePMD-kit-FastLearn/campaign_water_interface/deepmd/runs/deepmd_sB/s7/lcurve.out', usecols=(0,3,5))
deepmdles_sA = np.loadtxt('/root/app/deepmd-les/deepmd-les/DeePMD-kit-FastLearn/campaign_water_interface/deepmd/runs/deepmd-les_sA/s8/lcurve.out', usecols=(0,3,5))
deepmdles_sB = np.loadtxt('/root/app/deepmd-les/deepmd-les/DeePMD-kit-FastLearn/campaign_water_interface/deepmd/runs/deepmd-les_sB/s8/lcurve.out', usecols=(0,3,5))
cacesr_sA = np.loadtxt('/root/app/deepmd-les/deepmd-les/DeePMD-kit-FastLearn/campaign_water_interface/analysis/cace_sr_A_val_rmse.csv', usecols=(0,1,2),skiprows=1,delimiter=',')
cacesr_sB = np.loadtxt('/root/app/deepmd-les/deepmd-les/DeePMD-kit-FastLearn/campaign_water_interface/analysis/cace_sr_B_val_rmse.csv', usecols=(0,1,2),skiprows=1,delimiter=',')
cacelr_sA = np.loadtxt('/root/app/deepmd-les/deepmd-les/DeePMD-kit-FastLearn/campaign_water_interface/analysis/cace_lr_A_val_rmse.csv', usecols=(0,1,2),skiprows=1,delimiter=',')
cacelr_sB = np.loadtxt('/root/app/deepmd-les/deepmd-les/DeePMD-kit-FastLearn/campaign_water_interface/analysis/cace_lr_B_val_rmse.csv', usecols=(0,1,2),skiprows=1,delimiter=',')

seasr = np.loadtxt('/root/app/deepmd-les/deepmd-les/DeePMD-kit-FastLearn/campaign_fastlearn/cace/runs/cace-sea-sr/val_rmse.csv', usecols=(0,1,2),skiprows=1,delimiter=',')
sealr = np.loadtxt('/root/app/deepmd-les/deepmd-les/DeePMD-kit-FastLearn/campaign_fastlearn/cace/runs/cace-sea-lr/val_rmse.csv', usecols=(0,1,2),skiprows=1,delimiter=',')

arms = ['deepmdsr_sA', 'deepmdsr_sB', 'deepmdles_sA', 'deepmdles_sB', 'cacesr_sA', 'cacesr_sB', 'cacelr_sA', 'cacelr_sB', 'seasr', 'sealr']

for arm in arms[:4]:
    print(f'{arm}: rmse_e = {np.mean(eval(arm)[70:,1]):6f}, rmse_f = {np.mean(eval(arm)[70:,2]):6f}')
for arm in arms[4:]:
    print(f'{arm}: rmse_e = {np.mean(eval(arm)[-70:-1,1]):6f}, rmse_f = {np.mean(eval(arm)[-70:-1,2]):6f}')
