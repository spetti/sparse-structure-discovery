import numpy as np
import sys
import csv
import time

pos = np.array(sys.argv[1:]).astype('int')

snp_file = '/n/desai_lab/users/efenton/bbq/yeast_info/BYxRM_nanopore_SNPs.txt'
genes_file = '/n/desai_lab/users/efenton/bbq/yeast_info/BY4742_sorted_annotations.gff3'

with open(snp_file) as sf:
    for line in sf:
        snp_list = np.array(line.split('\t')).astype('float').astype('int')

pos = np.array(pos)
edges = np.array(np.where(snp_list == 2))
chro_idx = []
gene_names = []
lr_order = []

for p in pos:
    chro_idx.append(sum(sum(edges < p)) + 1)

pos_list = snp_list[pos + chro_idx - 1]

loc = []

with open(genes_file) as gf:
    next(gf)
    for line in gf:
        line = line.split('\t')
        for i in range(len(pos)):
            if int(line[0][3:]) == chro_idx[i] and int(line[3]) < pos_list[i] and int(line[4]) > pos_list[i]:
                if line[2] != 'CDS' and line[2] != 'contig':
                    gene_names.append(line[8].split(';')[1])
                else:
                    gene_names.append("--")
                lr_order.append(i)

x = np.argsort(pos)
print('Named loci:')
print(pos[lr_order])
print('Chromosome:')
print(chro_idx)
print('Position:')
print(pos_list[lr_order])
print('Gene names:')
print(gene_names)
