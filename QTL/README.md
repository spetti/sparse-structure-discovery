Description of files:

utils
- call_glmnet.py: python wrapper to call glmnet
- glmnet_fns.R: R code to call glmnet
- localize.py: localization code

jupyter notebooks
- process_BBQ_data.ipynb: designed to process BBQ genotype and phenotype data to form compatible with joint_qtl_mapping.py by
    - combining separate phenotype measurement files into one array
    - seperating genotype and phenotypes into training/val/test sets (80/10/10)
    - computing the correlation of nearby loci (using training set only)
    
- ci_to_orfs_and_annotations.ipynb: (TODO: test- for now just copy and pasted code) given confidence intervals, determines what orfs may be present and compiles annotation data frame
    
python scripts
- joint_qtl_mapping.py: performs all or part of the 5 step localization + optimization pipeline
    (1) greedy prefiltering: chooses a subset of loci to include so that no pair of loci is more than -ct correlated
    (2) glmnet round 1: runs glmnet on above subset of loci
    (3) localization round 1: for each loci with effect size > -lt1 in glmnet round 1 results, replace with more confident set of loci
    (4) glmnet round 2: runs glmnet on higher confidence set of loci
    (5) localization round 2: for each loci with effect size > -lt2 in glmnet round 2 results, find confidence interval across phenotypes

    Example usage for running the whole pipeline:
    
    python joint_qtl_mapping.py -gt <inloc/geno_train.npy> -pt <inloc/pheno_train.npy> -gv <inloc/geno_val.npy> -pv <inloc/pheno_val.npy> -c <inloc/cc_data_all.npy> -output <outloc> -v"
    
    Options to only perform part of pipeline:
    (Since runs of glmnet can be time intensive, we recommend these options for trying different parameters in steps 1,3,5 )
    - To only do (1) greedy prefiltering (for instance to see how many loci you end up with for a particular -ct), use -po
    - To skip (1) and instead input a list of loci to include using -l  (array of length number of loci in genotype; boolean to indicate whether to include)
    - To start at (3) localization round 1 use -sl1 (and -load_F to feed in F matrix and -l to feed in list of loci from the glmnet round 1 result); to only do (3) use -sl1o
    - To only do (5) localization round 2 use -sl2o (and -load_F to feed in F matrix and -l to feed in list of loci from the glmnet round 2 result)


- RUN_joint_qtl_mapping.py: example of how to run joint_qtl_mapping.py on Harvard's slurm cluster