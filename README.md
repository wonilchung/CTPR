# CTPR v1.1 User Manual 
#### Wonil Chung, January 29, 2019

## 1.	INTRODUCTION

The CTPR (Cross-Trait / Cross-eThnic Penalized Regression) software was originally developed for multi-trait polygenic risk prediction in large cohorts and is subsequently being extended for multi-ethnic polygenic risk prediction. It utilizes multiple secondary traits (or ethnicities) based on individual-level genotypes and/or summary statistics from large-scale GWAS studies to improve prediction accuracy. Based on penalized least squares methods, we propose a novel cross trait penalty function with the Lasso and the minimax concave penalty (MCP) to incorporate the shared genetic effects across multiple traits (or ethnicities) and implement it for large-sample GWAS data. Our approach extracts information from the secondary traits (or ethnicities) that is beneficial for predicting the primary trait (or ethnicity) but tunes down information that is not. Our novel implementation of a distributed memory parallel computing algorithm makes it feasible to apply our methods to biobank-scale GWAS data. We compared our multi-trait methods with other existing methods such as MTGBLUP (multi-trait genomic best linear unbiased prediction method), wMT-SBLUP (weighted multi-trait summary statistic best linear unbiased prediction method), MTAG (multi-trait analysis of GWAS) and showed that our approach outperforms them in predictive performance.

### Citations
The CTPR algorithm is described in the following reference:

Wonil Chung, Jun Chen, Constance Turman, Sara Lindstrom, Zhaozhong Zhu, Po-Ru Loh, Peter Kraft and Liming Liang (2019), Efficient cross-trait penalized regression increases prediction accuracy in large cohorts using secondary phenotypes. Nature Communications, 10, 569.

Wonil Chung and Liming Liang, (2019), Improving the polygenic risk prediction by incorporating LD information from multi-ethnic Biobank data. In preparation.

### Questions and Requests
If you have any questions on CTPR software, please email to Wonil Chung (wchung@hsph.harvard.edu).


## 2.	INSTALLING AND COMPILING CTPR

You can download the latest version of the CTPR software at: https://github.com/wonilchung/CTPR.

### 2.1	Change log

Version 1.2 (expected in February, 2019): 
-	Will support MACH dosage file format (e.g. test.mldose, test.mlinfo).
-	Will add options for data management to specify a list of individuals to be included or excluded and a list of SNPs to be included or exclude in the analysis.

Version 1.1 (January 29, 2019):
-	Modified file type for genotype data from double to float to decrease the memory size in half.
-	Added various options for data management.

Version 1.0 (March 3, 2017): 
-	Initial release of CTPR.

### 2.2	Installation

The CTPR_vX.X.tar.gz download package contains a standalone (i.e., statically linked) 64-bit Linux executable, CTPR, which we have tested on several Linux systems. If you wish to compile your own version of the CTPR software from the source code, you will need to ensure that compiler requirements and library dependencies are fulfilled, and you will need to make appropriate modifications to the Makefile (MakefileSpp for a single node version or MakfileMpi for MPI version). We explain how to install required packages and compile CTPR software on linux system below.

[CentOS]  

(1) install R and RcppArmadillo package  

wget http://mirror.las.iastate.edu/CRAN/src/base/R-3/R-3.4.4.tar.gz  
tar xvfz R-3.4.4.tar.gz  

install.packages('RcppArmadillo') 

(2) Install openmpi  

wget https://download.open-mpi.org/release/open-mpi/v2.0/openmpi-2.0.4.tar.gz  
tar -xvf openmpi-2.0.4.tar.gz  

(3) Compile CTPR  

make -f MakefileMpi  
make -f MakefileSpp  

[Debian Linux]

(1) Install R and RcppArmadillo package  

sudo apt-get update  
sudo apt-get install r-base-core  
sudo apt-get install r-base  

install.packages('RcppArmadillo')  

(2) Install openmpi  

apt-cache policy openmpi-bin  
apt-cache policy openmpi-doc  
apt-cache policy libopenmpi-dev  
apt-cache policy libibnetdisc-dev  
wget https://download.open-mpi.org/release/open-mpi/v2.0/openmpi-2.0.4.tar.gz  
sudo apt-get install libibnetdisc-dev  
tar -xvf openmpi-2.0.4.tar.gz  

(3) Compile CTPR  

make -f MakefileMpi  
make -f MakefileSpp  

### 2.3	Running CTPR

To run the ctpr executable, simply invoke ./ctpr or ./ctprmpi the Linux command line (within the CTPR install directory). The example/ subdirectory contains example data and code, so you can learn how to execute CTPR software. To obtain information on license of CTPR, run: ./ctpr –l or ./ctprmpi –l. To obtain full list of CTPR options, run: ./ctpr –h or ./ctprmpi –h.

## 3.	COMPUTING REQUIREMENTS

Basically, CTPR can run on any computing system including PC and Mac but with large-scale biobank-based GWAS data, we recommend using Linux-based high performance computing cluster. For distributed high performance computing, clusters utilize job schedulers such as LSF (Load Sharing Facility) and SLURM (Simple Linux Utility for Resource Management) to start, execute and monitor jobs on a set of allocated computing nodes. We will explain how to execute CTPR on cluster computer using SLURM.

### 3.1	Operating system

We have only compiled and tested CTPR on Linux computing environments including CenOS and Debian Linux. However, the source code is available and thus you can compile CTPR for a different OS.

### 3.2	Memory

Due to limited memory and computing resources, it may not be feasible to update the coefficients for all SNPs together. Alternatively, we propose to divide SNPs into multiple subgroups and allocate each subgroup to a MPI node for parallel computing. Each node updates only parameters in the subgroup with the remaining coefficients transferred from other nodes. Because MPI allows for communication between different nodes, the data of all nodes are synchronized at each estimation step.

### 3.3 Running Time

To assess the computational feasibility of CTPR for biobank-based GWAS data, we tested with N=437K individuals and P=1M SNPs from UK Biobank, which required ~1.7TB of memory with float data type (i.e. 437K*1M*4B=~1.7TB)4. The CTPR ran on 40 cores (Intel Xeon CPU 2.1 GHz) with 48GB of memory for each core, total of ~1.9TB of memory, for up to 7 days to complete the analyses with 40 core-groups (exact solution). The running time of CTPR depends linearly not only on the sample size (N) and the number of SNPs (P) but also on the number of core-group (q), which represents O(NPq). With 10 core-groups (approximate solution), the running time of CTPR dropped to ~1.75 days and it still generated almost the same predictive performance as exact solution due to good convergence. Even when sample size increases, the running time is able to remain similar because larger sample size increases likelihood of convergence and therefore less number of core-groups are needed. When you use Slurm scheduler, you can specify as

sbatch -n 40 --mem-per-cpu=48G -p mpi -t 5-00:00:00 --wrap="mpirun -np 40 ./ctprmpi --ng 40 …”

sbatch -n 40 --mem-per-cpu=48G -p mpi -t 5-00:00:00 --wrap="mpirun -np 40 ./ctprmpi --ng 10 …”


## 4.	INPUT FILE FORMAT

CTPR requires three input files containing genotypes, phenotypes, summary data. Genotype files can be in dosage (.dose) formats and phenotype and summary files can be in the text (txt) file.

### 4.1	Dosage File Format 

This dosage file contains genotype information. The number of rows is equivalent to the number of individuals and the number of columns is equal to the number of markers.

[genotype file]  
0.79304       0.54848      0.86099 ……..  
0.79304       -1.51711     -2.33368 ……..  
-0.87451     -1.51711      0.86099 ……..  

### 4.2	Phenotype File Format 

This file contains phenotype information. Each line is a number indicating the phenotype value for each individual in turn, in the same order as in genotype file and Each column is related to each phenotype. The number of rows should be equal to the number of individuals and the number of columns should be equal to the number of phenotype.

[phenotype file]  
-2.41873        -0.90627  
 0.74807        -0.20951  
 1.44223        1.26426  
……..

### 4.3	Summary File Format  

This file contains summary statistics for all markers. The first column is marker id, the second column is its minor allele frequency (MAF), the third column is its maker effect (β ̂) and the fourth column is its standard error of marker effect (se(β ̂)). This file contains marker effects for multiple phenotypes. The number of rows is the same as the number of markers (m) and the number of columns is 2*p+2 where p is the number of phenotypes. 

[summary file]   
1       0.43139     0.01826    0.01195  
2       0.67261    -0.00978    0.01205  
3       0.67029    -0.00969    0.01195  
4       0.33718     0.00069    0.01218  
……..  

## 5.	RUNNING CTPR

### 5.1	CTPR options 

--out or –output [prefix]: specify output file prefix  
--dos or –dosage [filename]: specify dosage file name for training  
--dos-ext or --dosage-extension [ext]: specify dosage file extension for training  
--phe or –phenotype [filename]: specify phenotype file name for training  
--dos-test or --dosage-test [filename]: specify dosage file name for testing  
--dos-test-ext or --dosage-test-extension [ext]: specify dosage file extension for testing  
--phe-test or --phenotype-test [filename]: specify phenotype file name for testing  
--sum or –summary [filename]: specify summary file name  
--sum-ext or --summary- extension [ext]: specify summary file extension  
--num-phe or --number-phenotype [num]: specify the number of phenotypes to be analyzed (default 1)  
--num-sum or --number-summary [num]: specify the number of phenotypes for summary file (default 1)  
--separ-ind or --separate-individual [num,num,..]: specify the numbers of individuals for each phenotypes separated by comma (,) in case of multiple phenotypes  
--penalty or --penalty-term [num]: specify the sparsity and cross-trait penalty terms (default 1; 1: Lasso+CTPR; 2: MCP+CTPR)  
--nfold or --number-fold [num]: specify the number of folds for coordinate decent algorithm (default 5)  
--prop or –proportion [num]: specify proportion of maximum number of non-zero beta (default 0.25)  
--lambda2 or --lambda2-option [num]: specify value for lambda2. If negative value is specified, pre-specified values are used for lambda2 (default -3; -1: λ_2=(0, 0.94230, 1.60280, 3.37931); -2: λ_2=(0, 0.06109, 0.94230, 0.13920, 0.24257, 0.38582, 0.59756, 0.94230, 1.60280, 3.37931); -3: λ_2=(0, 0.06109, 0.94230, 0.13920, 0.24257, 0.38582, 0.59756, 0.94230, 1.60280, 3.37931, 24.5); -4: λ_2=(0, 0.06109, 0.94230, 0.13920, 0.24257, 0.38582, 0.59756, 0.94230, 1.60280, 3.37931, 8.5, 15.5, 24.5))  
--ng or --number-group [num]: specify the number of group for MPI mode (default number of MPI nodes)  
--st or --start-number [num]: specify starting number for MPI files (default 1)  
--flamb1 or --first-lambda1 [num]: specify first lambda1 value for MPI mode (default 1)  
--llamb1 or --last-lambda1 [num]: specify last lambda1 value for MPI mode (default 100)  
--scaling or --scaling-phenotype: specify for scaling secondary phenotypes using simple linear regression between phenotypes and genotypes  

### 5.2	Example codes 

[running ctpr]
 
./ctpr \  
  --out  ./res/test \  
  --dos final_5000_train.dose \  
  --phe final_pheno_5000_train.phe \  
  --dos-test final_5000_test.dose \  
  --phe-test final_5000_test.phe \  
  --separ-ind 7400,7400 \  
  --penalty 1 \  
  --lambda2 0  

[running ctprmpi]
  
mpirun -x LD_PRELOAD=libmpi.so -np 2 ./ctprmpi \  
  --out  ./res/testmpimcp \  
  --dos final_one_scaled_5000_train \  
  --dos-ext dose \  
  --phe final_one_summary_pheno_5000_train.phe \  
  --dos-test final_one_scaled_5000_test \  
  --dos-test-ext dose \  
  --phe-test final_one_summary_pheno _5000_test.phe \  
  --sum final_one_marginal_beta _part \  
  --sum-ext txt \  
  --num-phe 1 \  
  --penalty 2 \  
  --lambda2 0.13920  


