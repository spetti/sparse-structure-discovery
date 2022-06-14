#! /usr/bin/python
import subprocess
import os
from subprocess import call

date = "6_14"
#date = "6_14_lt1_0"
inloc = "BBQ_data_processed"
outloc = f"BBQ_results_{date}"
try:
    os.mkdir(outloc)
except:
    print(f"directory {outloc} exists")

name = f"joint_qtl"
batchfileName = f"/tmp/{date}_{name}"
batchfile = open(batchfileName, "w")
batchfile.write("#!/bin/bash \n")
batchfile.write("#SBATCH -c 25 \n")
batchfile.write("#SBATCH -t 1-00:00 \n")
batchfile.write("#SBATCH -p eddy \n")
batchfile.write("#SBATCH --mem=100000 \n")
batchfile.write(f"#SBATCH -o ./slurm/out.{name} \n")
batchfile.write(f"#SBATCH -e ./slurm/err.{name} \n")
batchfile.write("#SBATCH --mail-type=END \n")

batchfile.write("module load R/4.0.2-fasrc01")
batchfile.write("\n")

batchfile.write("source activate jupyter_R")
batchfile.write("\n")


commandstring=f"python joint_qtl_mapping.py -gt {inloc}/geno_train.npy -pt {inloc}/pheno_train.npy -gv {inloc}/geno_val.npy -pv {inloc}/pheno_val.npy -c {inloc}/cc_data_all.npy -output {outloc} -v"

#print(commandstring)
batchfile.write(commandstring)
batchfile.write("\n")
batchfile.close()

sbatchstring =["sbatch", batchfileName]
call(sbatchstring)
os.remove(batchfileName)