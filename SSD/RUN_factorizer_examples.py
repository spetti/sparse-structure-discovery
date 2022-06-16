#! /usr/bin/python
import subprocess
import os
from subprocess import call

out_location = "pickled_factorizers_6_15"

for mode in ["bbq"]: #, "syn_hub","bbq", "kinsler", "genotoxin"]:
    ext = f"{mode}"
    batchfileName = f"/tmp/{ext}"
    batchfile = open(batchfileName, "w")
    batchfile.write("#!/bin/bash \n")
    batchfile.write("#SBATCH -c 4 \n")
    batchfile.write("#SBATCH -t 2-00:00 \n")
    batchfile.write("#SBATCH -p eddy \n")
    batchfile.write("#SBATCH --mem=16000 \n")
    batchfile.write(f"#SBATCH -o ./slurm/out.{ext} \n")
    batchfile.write(f"#SBATCH -e ./slurm/err.{ext} \n")
    batchfile.write("#SBATCH --mail-type=END \n")

    batchfile.write("module load R/4.0.2-fasrc01")
    batchfile.write("\n")

    batchfile.write("source activate jupyter_3.6")
    batchfile.write("\n")

    commandstring=f"python factorizer_examples.py {mode} {out_location}"

    batchfile.write(commandstring)
    batchfile.write("\n")
    batchfile.close()

    sbatchstring =["sbatch", batchfileName]
    call(sbatchstring)
    os.remove(batchfileName)