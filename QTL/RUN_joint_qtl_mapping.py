#! /usr/bin/python
import subprocess
import os
from subprocess import call
#for pair in [("l1", "0.0"),("l1", "0.001"),("l2", "0.001")]:

#for pair in [("l1", "0.0043")]:
    #, ("l2","0.003")]:
#for pair in [("l1", "0.0"),("l1", "0.001"),("l1", "0.0015"),("l2", "0.001"), ("l2", "0.0015")]:
    #norm, lt1  = pair[0],pair[1]
width = 100
for ct in [0.94, 0.97]:
    for pair in [("l1", "0.0"),("l1", "0.001"),("l1", "0.002"),("l1", "0.003"), ("l2", "0.001"), ("l2", "0.002"), ("l2", "0.003")]:
        norm, lt1  = pair[0],pair[1]
        date = f"6_18_{ct}_{norm}_{lt1}"
        inloc = "BBQ_data_processed"
        outloc = f"BBQ_results_6_18"
        try:
            os.mkdir(outloc)
        except:
            print(f"directory {outloc} exists")

        name = f"qtl_{date}"
        batchfileName = f"/tmp/{name}"
        batchfile = open(batchfileName, "w")
        batchfile.write("#!/bin/bash \n")
        batchfile.write("#SBATCH -c 25 \n")
        batchfile.write("#SBATCH -t 4:00:00 \n")
        batchfile.write("#SBATCH -p eddy \n")
        batchfile.write("#SBATCH --mem=100000 \n")
        batchfile.write(f"#SBATCH -o ./slurm/out.{name} \n")
        batchfile.write(f"#SBATCH -e ./slurm/err.{name} \n")
        batchfile.write("#SBATCH --mail-type=END \n")

        batchfile.write("module load R/4.0.2-fasrc01")
        batchfile.write("\n")

        batchfile.write("source activate jupyter_R")
        batchfile.write("\n")


        #commandstring=f"python joint_qtl_mapping.py -gt {inloc}/geno_train.npy -pt {inloc}/pheno_train.npy -gv {inloc}/geno_val.npy -pv {inloc}/pheno_val.npy -c {inloc}/cc_data_all.npy -output {outloc} -v -ct {ct}"

        commandstring=f"python joint_qtl_mapping.py -gt {inloc}/geno_train.npy -pt {inloc}/pheno_train.npy -gv {inloc}/geno_val.npy -pv {inloc}/pheno_val.npy -output {outloc} -v -l {outloc}/loci_kept_cc_{ct}.npy -load_F {outloc}/first_F_cc_{ct}.npy -load_P {outloc}/first_preds_cc_{ct}.npy -sl1 -norm {norm} -lt1 {lt1} -w {width} -ct {ct}"

        #print(commandstring)
        batchfile.write(commandstring)
        batchfile.write("\n")
        batchfile.close()

        sbatchstring =["sbatch", batchfileName]
        call(sbatchstring)
        os.remove(batchfileName)