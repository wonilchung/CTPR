/******************************************************************************
* File: CTPRSrcFunc.cpp
* Version: CTPR 1.1
* Author: Wonil Chung, Jun Chen
* First Revised: 02.14.2014
* Last Revised:  01.29.2019
* Description: C++ Functions for Cross-Trait Penalized Regression 
******************************************************************************/


#include <RcppArmadillo.h>
#include <Rcpp.h>
#include <unistd.h>
#include <string.h> 
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <algorithm>

using namespace Rcpp;


/*************************************************************************/
/*                                                                       */
/* Class for All Paramters                                              */
/*                                                                       */
/*************************************************************************/

class PARAM {
public:
	
	// For input data
	std::string fnameXs, fextXs, fnameXstest, fextXstest;
	std::string fnameSs, fextSs;
	std::string fnameYs, fnameYstest;
	std::string penalize, penalize2, output, ranktxt;
	arma::fvec nrowXs, nrowYs; int nrowXstest, nrowYstest;
	int ncolXs, ncolYs, ncolXstest, ncolYstest;
	int useSummary, useTest, nsecTrait, separinds, nFold; 
	int nGroup, start, flamb1, llamb1, rank, size, useScaling;
	float perc;
	clock_t t_start;
	
	// For Internal use
	int useMPI, maxNz, maxIter;
	bool error, re_est, slambda2;
	float gamma, tol;
	arma::field<arma::fvec> lambda1;
	arma::fvec lambda2;
	arma::umat nzero;

};


/*************************************************************************/
/*                                                                       */
/* Functions for String                                                  */
/*                                                                       */
/*************************************************************************/

// to_string function
namespace patch {
	template < typename T > std::string to_string( const T& n ){
		std::ostringstream stm ;
		stm << n ;
		return stm.str() ;
	}
}

// Split string 
std::vector<std::string> split(std::string str, std::string sep){
	char* cstr=const_cast<char*>(str.c_str());
	char* current;
	std::vector<std::string> arr;
	current=strtok(cstr,sep.c_str());
	while(current!=NULL){
		arr.push_back(current);
		current=strtok(NULL,sep.c_str());
	}
	return arr;
}

// Trim string
void trim(std::string& s) {
	std::size_t p = s.find_first_not_of(" \t");
	s.erase(0, p);
	p = s.find_last_not_of(" \t\r");
	if (std::string::npos != p) s.erase(p+1);
}


/*************************************************************************/
/*                                                                       */
/* Functions for Primary Check                                           */
/*                                                                       */
/*************************************************************************/


std::string version = "1.1";
std::string date = "January 29 2019";
std::string year = "2017-2019";

void PrintVersion(){
	std::cout << "**************************************************************" << std::endl;
	std::cout << "* CTPR v" << version << " "<< date << "                       " <<std::endl;
	std::cout << "* Cross-Trait Penalized Regression                            " << std::endl;
	std::cout << "* Cross-eThnic Penalized Regression                           " << std::endl;
	std::cout << "* Developed by Wonil Chung                                    " << std::endl;
	std::cout << "**************************************************************" << std::endl;
	std::cout << std::endl;
	std::cout << "Visit https://github.com/wonilchung/CTPR for latest updates   " << std::endl;
	std::cout << "Copyright (C) " << year << " Wonil Chung                     " << std::endl;
	std::cout << "For Help, Type ./ctpr -h  or ./ctprmpi -h                    " << std::endl;
	std::cout << std::endl;
}

void PrintLicense(){
	PrintVersion();
	std::cout << "CTPR Software is freely available to only academic users.    " << std::endl;
	std::cout << std::endl;
}

void PrintHelp(){
	PrintVersion();
	std::cout << "CTPR Options: " << std::endl;
	std::cout << "  --out or --output [prefix]: specify output file prefix" << std::endl;
	std::cout << "  --dos or --dosage [filename]: specify dosage file name for training" << std::endl;
	std::cout << "  --dos-ext or --dosage-extension [ext]: specify dosage file extension for training" << std::endl;
	std::cout << "  --phe or --phenotype [filename]: specify phenotype file name for training" << std::endl;
	std::cout << "  --dos-test or --dosage-test [filename]: specify dosage file name for testing" << std::endl;
	std::cout << "  --dos-test-ext or --dosage-test-extension [ext]: specify dosage file extension for testing" << std::endl;
	std::cout << "  --phe-test or --phenotype-test [filename]: specify phenotype file name for testing" << std::endl;
	std::cout << "  --sum or --summary [filename]: specify summary file name" << std::endl;
	std::cout << "  --sum-ext or --summary-extension [ext]: specify summary file extension" << std::endl;
	std::cout << "  --num-phe or --number-phenotype [num]: specify the number of phenotypes to be analyzed (default 1)" << std::endl;
	std::cout << "  --num-sum or --number-summary [num]: specify the number of phenotypes for summary file (default 1)" << std::endl;
	std::cout << "  --separ-ind or --separate-individual [num,num,..]: specify the numbers of individuals for each phenotypes separated by comma (,) in case of multiple phenotypes" << std::endl;
	std::cout << "  --penalty or --penalty-term [num]: specify the sparsity and cross-trait penalty terms (default 1; 1: Lasso+CTPR; 2: MCP+CTPR)" << std::endl;
	std::cout << "  --nfold or --number-fold [num]: specify the number of folds for coordinate decent algorithm (default 5)" << std::endl;
	std::cout << "  --prop or --proportion [num]: specify proportion of maximum number of non-zero beta (default 0.25)" << std::endl;
	std::cout << "  --lambda2 or --lambda2-option [num]: specify value for lambda2" << std::endl;
	std::cout << "    If negative value is specified, pre-specified values are used for lambda2 (default -3)" << std::endl;
	std::cout << "    -1: lambda2=(0, 0.94230, 1.60280, 3.37931)" << std::endl;
	std::cout << "    -2: lambda2=(0, 0.06109, 0.94230, 0.13920, 0.24257, 0.38582, 0.59756, 0.94230, 1.60280, 3.37931)" << std::endl;
	std::cout << "    -3: lambda2=(0, 0.06109, 0.94230, 0.13920, 0.24257, 0.38582, 0.59756, 0.94230, 1.60280, 3.37931, 24.5)" << std::endl;
	std::cout << "    -4: lambda2=(0, 0.06109, 0.94230, 0.13920, 0.24257, 0.38582, 0.59756, 0.94230, 1.60280, 3.37931, 8.5, 15.5, 24.5)" << std::endl;
	std::cout << "  --ng or --number-group [num]: specify the number of group for MPI mode (default number of MPI nodes)" << std::endl;
	std::cout << "  --st or --start-number [num]: specify starting number for MPI files (default 1)" << std::endl;
	std::cout << "  --flamb1 or --first-lambda1 [num]: specify first lambda1 value for MPI mode (default 1)" << std::endl;
	std::cout << "  --llamb1 or --last-lambda1 [num]: specify last lambda1 value for MPI mode (default 100)" << std::endl;
	std::cout << "  --scaling or --scaling-phenotype: specify for scaling secondary phenotypes using simple linear regression between phenotypes and genotypes" << std::endl;
	std::cout << std::endl;
}

void PrintCurrentTime(PARAM &cPar){
	char buff[100]; 
	time_t now; 
	clock_t t_current;
	int seconds, minutes, hours;

	now = time(0); 
	strftime(buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime(&now));
	t_current = clock(); 
	seconds = (int) (t_current)/CLOCKS_PER_SEC; 
	minutes = (int) seconds / 60;
	seconds %= 60;
	hours = (int) minutes / 60;
	minutes %= 60;
	if (cPar.rank==0) std::cout << "Current Time: " << buff << ", CPU time used: " << hours << " hours, " << minutes << " minutes, " << seconds << " seconds"  << std::endl;
	std::cout << std::endl;
}

int AssignParameters(int argc, char **argv, PARAM &cPar){
	std::cout << "**************************************************************" << std::endl;
	std::cout << "                    PARAMETER ASSIGNMENT                      " << std::endl;
	std::cout << "**************************************************************" << std::endl;
	std::cout << std::endl;

	std::string str;
	std::vector<std::string> vec;
	int i=0, j=0, tmp=0;
	float tmpf = 0;

	std::cout << "Command Line Options:" << cPar.ranktxt << std::endl;
	for(i = 1; i < argc; i++) { 
		if (std::strcmp(argv[i], "--out")==0 || std::strcmp(argv[i], "--output")==0) { // output file
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.output = str;
			std::cout << " Output File: " << str << cPar.ranktxt << std::endl;
		}
		else if (std::strcmp(argv[i], "--dos")==0 || std::strcmp(argv[i], "--dosage")==0) { // dosage file
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.fnameXs = str;
			std::cout << " Genotype File for Training: " << str << cPar.ranktxt << std::endl;
		}
		else if (std::strcmp(argv[i], "--dos-ext")==0 || std::strcmp(argv[i], "--dosage-extension")==0) { // dosage extension
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.fextXs = str;
			std::cout << " Genotype File Extension for Training: " << str << cPar.ranktxt << std::endl;
		}
		else if (std::strcmp(argv[i], "--sum")==0 || std::strcmp(argv[i], "--summary")==0) { // summary statistics
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.fnameSs = str;
			cPar.useSummary = 1; ////
			std::cout << " Summary Data File: " << str << cPar.ranktxt << std::endl;
		}
		else if (std::strcmp(argv[i], "--sum-ext")==0 || std::strcmp(argv[i], "--summary-extension")==0) { // summary extension
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.fextSs = str;
			std::cout << " Summary Data File Extension: " << str << cPar.ranktxt << std::endl;
		}
		else if (std::strcmp(argv[i], "--phe")==0 || std::strcmp(argv[i], "--phenotype")==0) { // phenotype file
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.fnameYs = str;
			std::cout << " Phenotype File: " << str << cPar.ranktxt << std::endl;
		}
		else if (std::strcmp(argv[i], "--dos-test")==0 || std::strcmp(argv[i], "--dosage-test")==0) { // dosage file for testing
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.fnameXstest = str;
			cPar.useTest = 1; ////
			std::cout << " Genotype File for Testing: " << str << cPar.ranktxt << std::endl;
		}
		else if (std::strcmp(argv[i], "--dos-test-ext")==0 || std::strcmp(argv[i], "--dosage-test-extension")==0) { // dosage extension for testing
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.fextXstest = str;
			std::cout << " Genotype File Extention for Testing: " << str << cPar.ranktxt << std::endl;
		}
		else if (std::strcmp(argv[i], "--phe-test")==0 || std::strcmp(argv[i], "--phenotype-test")==0) { // phenotype file for testing
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.fnameYstest = str;
			cPar.useTest = 1; ////
			std::cout << " Phenotype File for Testing: " << str << cPar.ranktxt << std::endl;
		}
		else if (std::strcmp(argv[i], "--num-phe")==0 || std::strcmp(argv[i], "--number-phenotype")==0) { // # phenotypes to be analyzed
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.ncolYs = atoi(str.c_str());
			std::cout << " Number of Phenotypes to be Analyzed: " << str << cPar.ranktxt << std::endl;
			if (cPar.ncolYs < 1) { cPar.error = TRUE; std::cout << " An error Occurred on --num-phe or --number-phenotype..." << std::endl; break; }
		}
		else if (std::strcmp(argv[i], "--num-sum")==0 || std::strcmp(argv[i], "--number-summary")==0) { // # phenotypes for summary statistics
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.nsecTrait = atoi(str.c_str());
			std::cout << " Number of Phenotypes for Summary Data to be Analyzed: " << str << cPar.ranktxt << std::endl;
			if (cPar.nsecTrait < 1) { cPar.error = TRUE; std::cout << " An error Occurred on --num-sum or --number-summary..." << std::endl; break; }
		}
		else if (std::strcmp(argv[i], "--separ-ind")==0 || std::strcmp(argv[i], "--separate-individual")==0) { // vector for # individuals
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			vec = split(str, ",");  //for (j=0; j<vec.size(); j++) std::cout << "j" << vec[j] << cPar.ranktxt << std::endl;
			cPar.nrowXs.set_size(vec.size()+1); cPar.nrowXs(0)=0;
			cPar.nrowYs.set_size(vec.size()+1); cPar.nrowYs(0)=0;
			for (j=0; j<vec.size(); j++) cPar.nrowXs(j+1) = cPar.nrowXs(j) + atoi(vec[j].c_str());
			for (j=0; j<vec.size(); j++) cPar.nrowYs(j+1) = cPar.nrowYs(j) + atoi(vec[j].c_str());
			cPar.separinds = 1; // Set separinds to 1 to use separate individuals for each phenotype
			std::cout << " Number of Individiuals for Each Phenotype: ";
			for (j=0; j<vec.size(); j++) std::cout << vec[j].c_str() << " "; std::cout << cPar.ranktxt << std::endl;
			for (j=0; j<vec.size(); j++) if (atoi(vec[j].c_str())<1) { cPar.error = TRUE; std::cout << " An error Occurred on --separ-ind or --separate-individual..." << std::endl; break; }
		}
		else if (std::strcmp(argv[i], "--penalty")==0 || std::strcmp(argv[i], "--penalty-term")==0) { // choice for penalty terms
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			tmp = atoi(str.c_str());
			if (tmp < 1 || tmp > 2) { cPar.error = TRUE; std::cout << " An error Occurred on -penalty..." << std::endl; break; }
			if (tmp == 1) { cPar.penalize  = "Lasso"; cPar.penalize2 = "CTPR"; } // Lasso 
			if (tmp == 2) { cPar.penalize  = "MCP"; cPar.penalize2 = "CTPR"; } // MCP 
			std::cout << " Penalty Terms: " << cPar.penalize << "+" << cPar.penalize2 << cPar.ranktxt << std::endl;
		}
		else if (std::strcmp(argv[i], "--nfold")==0 || std::strcmp(argv[i], "--number-fold")==0) { // number of folds
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.nFold = atoi(str.c_str());
			std::cout << " Number of Folds: " << str << cPar.ranktxt << std::endl;
			if (cPar.nFold < 1) { cPar.error = TRUE; std::cout << " An error Occurred on --nfold or --number-fold..." << std::endl; break; }
		}
		else if (std::strcmp(argv[i], "--prop")==0 || std::strcmp(argv[i], "--proportion")==0) { // proportion of maximum number of non-zero beta
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.perc = atof(str.c_str());
			std::cout << " Proportion of Maximum Number of Non-zero Beta: " << str << cPar.ranktxt << std::endl;
			if (cPar.perc <= 0 || cPar.perc > 1) { cPar.error = TRUE; std::cout << " An error Occurred on --prop or --proportion..." << std::endl; break; }
		}
		else if (std::strcmp(argv[i], "--lambda2")==0 || std::strcmp(argv[i], "--lambda2-option")==0) { // option for lambda2 vector 
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			tmpf = atof(str.c_str());
			if (tmpf < 0 && (tmpf != -1 && tmpf != -2 && tmpf != -3 && tmpf != -4)) { cPar.error = TRUE; std::cout << " An error Occurred on --lambda2 or --lambda2-option..." << std::endl; break; }
			cPar.slambda2 = TRUE; ////
			if (tmpf >= 0) {
				cPar.lambda2.set_size(1); cPar.lambda2(0)=tmpf;
			}
			else if (tmpf == -1) {
				cPar.lambda2.set_size(4);
				cPar.lambda2(0)=0; cPar.lambda2(1)=0.94230; cPar.lambda2(2)=1.60280; cPar.lambda2(3)=3.37931; 
			}
			else if (tmpf == -2){
				cPar.lambda2.set_size(9); 
				cPar.lambda2(0)=0; cPar.lambda2(1)=0.06109; cPar.lambda2(2)=0.13920 ; cPar.lambda2(3)= 0.24257; cPar.lambda2(4)=0.38582;
				cPar.lambda2(5)=0.59756; cPar.lambda2(6)=0.94230; cPar.lambda2(7)=1.60280; cPar.lambda2(8)=3.37931;
			}
			else if (tmpf == -3) {
				cPar.lambda2.set_size(10);
				cPar.lambda2(0)=0; cPar.lambda2(1)=0.06109; cPar.lambda2(2)=0.13920 ; cPar.lambda2(3)= 0.24257; cPar.lambda2(4)=0.38582;
				cPar.lambda2(5)=0.59756; cPar.lambda2(6)=0.94230; cPar.lambda2(7)=1.60280; cPar.lambda2(8)=3.37931; cPar.lambda2(9)=24.5;
			}
			else if (tmpf == -4) {
				cPar.lambda2.set_size(12);
				cPar.lambda2(0)=0; cPar.lambda2(1)=0.06109; cPar.lambda2(2)=0.13920 ; cPar.lambda2(3)= 0.24257; cPar.lambda2(4)=0.38582;
				cPar.lambda2(5)=0.59756; cPar.lambda2(6)=0.94230; cPar.lambda2(7)=1.60280; cPar.lambda2(8)=3.37931; cPar.lambda2(9)=8.5;  
				cPar.lambda2(10)=15.5; cPar.lambda2(11)=24.5;
			}
			std::cout << " Lambda2 for cross-trait penalty: ";
			for (j=0; j<cPar.lambda2.n_elem; j++) std::cout << cPar.lambda2(j) << " ";
			std::cout << cPar.ranktxt << std::endl;
		}
		else if (std::strcmp(argv[i], "--ng")==0 || std::strcmp(argv[i], "--number-group")==0) { // # group for MPI
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.nGroup = atoi(str.c_str());
			std::cout << " Number of Group for MPI: " << str << cPar.ranktxt << std::endl;
			if (cPar.nGroup < 1) { cPar.error = TRUE; std::cout << " An error Occurred on --ng or --num-group..." << std::endl; break; }
		}
		else if (std::strcmp(argv[i], "--st")==0 || std::strcmp(argv[i], "--start-number")==0) { // starting number for MPI files
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.start = atoi(str.c_str());
			std::cout << " Starting number for MPI files: " << str << cPar.ranktxt << std::endl;
			if (cPar.start < 0) { cPar.error = TRUE; std::cout << " An error Occurred on --st or --start-number..." << std::endl; break; }
		}
		else if (std::strcmp(argv[i], "--flamb1")==0 || std::strcmp(argv[i], "--first-lambda1")==0) { // first lambda1
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.flamb1 = atoi(str.c_str());
			std::cout << " First lambda1: " << str << cPar.ranktxt << std::endl;
			if (cPar.flamb1 < 0 || cPar.flamb1 > 100) { cPar.error = TRUE; std::cout << " An error Occurred on --flamb1 or --first-lambda1..." << std::endl; break; }
		}
		else if (std::strcmp(argv[i], "--llamb1")==0 || std::strcmp(argv[i], "--last-lambda1")==0) { // last lambda1
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.llamb1 = atoi(str.c_str());
			std::cout << " Last lambda1: " << str << cPar.ranktxt << std::endl;
			if (cPar.llamb1 < 0 || cPar.llamb1 > 100) { cPar.error = TRUE; std::cout << " An error Occurred on --llamb1 or --last-lambda1..." << std::endl; break; }
		}
		else if (std::strcmp(argv[i], "--scaling")==0 || std::strcmp(argv[i], "--scaling-phenotype")==0) { // for scaling secondary phenotypes
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			cPar.useScaling = 1;
			std::cout << " Conducting Linear Regression for Scaling Secondary Traits..." << std::endl;
		}
		else {std::cout << " An error Occurred on Unrecognized Option: " << argv[i] << cPar.ranktxt << std::endl; cPar.error = TRUE; break;}
	}

	// Check there are any errors
	if (cPar.error) { return 0; }

	std::cout << std::endl;

	return 1;
}

int ScaleTraits(arma::field<arma::fvec> & Ys, arma::field<arma::fmat> & Xs, std::string szYs, PARAM &cPar){
	std::cout << "**************************************************************" << std::endl;
	std::cout << "                SCALING SECONDARY TRAITS                      " << std::endl;
	std::cout << "**************************************************************" << std::endl;
	std::cout << std::endl;
	
	int n = Ys(0).n_elem;
	int m = Ys.n_elem;
	int p = Xs(0).n_cols;

	return 1;
}

int CheckCombination(PARAM &cPar){
	std::cout << "**************************************************************" << std::endl;
	std::cout << "              PARAMETER COMBINATION CHECK                     " << std::endl;
	std::cout << "**************************************************************" << std::endl;
	std::cout << std::endl;

	if (cPar.flamb1 > cPar.llamb1) {
		std::cout << "Error! first lambda1 is larger than last lambda1. Please check it again..." << cPar.ranktxt << std::endl;
		return 0;
	}

	return 1;
}


/*************************************************************************/
/*                                                                       */
/* Functions for File IO                                                 */
/*                                                                       */
/*************************************************************************/

// Check All Input Files
int CheckData(std::string szXs, std::string szXstest, std::string szYs, std::string szYstest, std::string szSs, PARAM & cPar){
	std::cout << "**************************************************************" << std::endl;
	std::cout << "       PRELIMINARY PHENOTYPE/GENOTYPE/SUMMARY FILE CHECK      " << std::endl;
	std::cout << "**************************************************************" << std::endl;
	std::cout << std::endl;

	// Check Genotype and Phenotype data
	std::fstream fs;
	int rows=0, cols=0, numItems=0;
	std::string val;

	std::cout << "Performing basic file check on training genotype data :  " << szXs << cPar.ranktxt << std::endl;
	std::cout << "Checking file..." << cPar.ranktxt << std::endl;
	fs.open(szXs.c_str(), std::fstream::in);
	if (fs.is_open()) {
		while( fs.peek() != '\n' && fs >> val ){ ++numItems; }
		cols = numItems; numItems = 0;// # of columns 
		while( fs.peek() != EOF ){ getline(fs, val); ++numItems; }
		rows = numItems; // # of rows
		std::cout << rows << " samples and " << cols << " markers found in the file..." << cPar.ranktxt << std::endl;
		cPar.ncolXs = cols; // Set # markers always ///////////////
		if (!cPar.separinds) cPar.nrowXs(1) = rows; // Set nrowXs when using overlapping samples ////////////
	} else {
		std::cout << "Please check the path of training genotype data : " << szXs << cPar.ranktxt << std::endl;
		return 0;
	}
	fs.close();
	std::cout << std::endl;

	numItems = 0;
	std::cout << "Performing basic file check on training phenotype data :  " << szYs << cPar.ranktxt << std::endl;
	std::cout << "Checking file..." << cPar.ranktxt << std::endl;
	fs.open(szYs.c_str(), std::fstream::in);
	if (fs.is_open()) {
		while( fs.peek() != '\n' && fs >> val ){ ++numItems; }
		cols = numItems; numItems = 0;// # of columns 
		while( fs.peek() != EOF ){ getline(fs, val); ++numItems; }
		rows = numItems; // # of rows
		std::cout << rows << " samples and " << cols << " phenotypes found in the file..." << cPar.ranktxt << std::endl;
		if (cPar.ncolYs == 0) cPar.ncolYs = cols; // Set # phenotypes only when ncolYs is not specified /////////
		if (!cPar.separinds) cPar.nrowYs(1) = rows; // Set nrowYs when using overlapping samples ////////////
	} else {
		std::cout << "Please check the path of training phenotype data : " << szYs << cPar.ranktxt << std::endl;
		return 0;
	}
	fs.close();
	std::cout << std::endl;

	if (cPar.useTest) {
		numItems = 0;
		std::cout << "Performing basic file check on testing genotype data :  " << szXstest << cPar.ranktxt << std::endl;
		std::cout << "Checking file..." << cPar.ranktxt << std::endl;
		fs.open(szXstest.c_str(), std::fstream::in);
		if (fs.is_open()) {
			while( fs.peek() != '\n' && fs >> val ){ ++numItems; }
			std::cout << std::endl;
			cols = numItems; numItems = 0;// # of columns 
			while( fs.peek() != EOF ){ getline(fs, val); ++numItems; }
			rows = numItems; // # of rows
			std::cout << rows << " samples and " << cols << " markers found in the file..." << cPar.ranktxt << std::endl;
			cPar.ncolXstest = cols; // Set # markers always ///////////////
			cPar.nrowXstest = rows; // Set # rows always //////////////
			
		} else {
			std::cout << "Please check the path of testing genotype data : " << szXstest << cPar.ranktxt << std::endl;
			return 0;
		}
		fs.close();
		std::cout << std::endl;
		
		numItems = 0;
		std::cout << "Performing basic file check on testing phenotype data :  " << szYstest << cPar.ranktxt << std::endl;
		std::cout << "Checking file..." << cPar.ranktxt << std::endl;
		fs.open(szYstest.c_str(), std::fstream::in);
		if (fs.is_open()) {
			while( fs.peek() != '\n' && fs >> val ){ ++numItems; }
			cols = numItems; numItems = 0;// # of columns 
			while( fs.peek() != EOF ){ getline(fs, val); ++numItems; }
			rows = numItems; // # of rows
			std::cout << rows << " samples and " << cols << " phenotypes found in the file..." << cPar.ranktxt << std::endl;
			cPar.ncolYstest = cols; // Set # phenotypes always /////////
			cPar.nrowYstest = rows; // Set # rows always ////////////////
		} else {
			std::cout << "Please check the path of testing phenotype data : " << szYstest << cPar.ranktxt << std::endl;
			return 0;
		}
		fs.close();
		std::cout << std::endl;
	}
	
	if (cPar.useSummary) {
		numItems = 0;
		std::cout << "Performing basic file check on summary data :  " << szSs << cPar.ranktxt << std::endl;
		std::cout << "Checking file..." << cPar.ranktxt << std::endl;
		fs.open(szSs.c_str(), std::fstream::in);
		if (fs.is_open()) {
			while( fs.peek() != '\n' && fs >> val ){ ++numItems; }
			cols = numItems; numItems = 0;// # of columns 
			while( fs.peek() != EOF ){ getline(fs, val); ++numItems; }
			rows = numItems; // # of rows
			if (cols < 4 || (cols-2)%2!=0) {std::cout << "Please double-check summary data..." << cPar.ranktxt << std::endl; return 0;}
			cols = (int)(cols-2)/2;
			if (cPar.nsecTrait == 0) cPar.nsecTrait = cols;
			std::cout << rows << " markers and " << cols << " phenotypes found in the file..." << cPar.ranktxt << std::endl;
		} else {
			std::cout << "Please check the path of summary data : " << szSs << cPar.ranktxt << std::endl;
			return 0;
		}
		fs.close();
		std::cout << std::endl;
	}

	cPar.maxNz = (int)cPar.ncolXs*cPar.perc; 
	std::cout << "Maximum Number of Non-zero beta: " << cPar.maxNz << cPar.ranktxt << std::endl;

	std::cout << "Initial basic file check on phenotype/genoytype/summary data successful !!!" << cPar.ranktxt << std::endl;
	std::cout << std::endl;
	
	return 1;
}

// Read Phenotype File
void LoadPheno(arma::field<arma::fvec> & Ys, arma::fvec & Ystest, std::string szYs, std::string szYstest, PARAM & cPar) {
	int i, k;
	std::fstream fs;
	std::string line;
	std::vector<std::string> vec;

	std::cout << "**************************************************************" << std::endl;
	std::cout << "                   READING PHENOTYPE FILES                    " << std::endl;
	std::cout << "**************************************************************" << std::endl;
	std::cout << std::endl;
	
	// Read phenotype data for training 
	fs.open(szYs.c_str(), std::fstream::in);
	if (cPar.separinds) { // no overlapping samples for multiple phenotypes
		for (i=0; i<cPar.nrowYs.n_elem-1; i++) Ys(i) = arma::zeros<arma::fvec>(cPar.nrowYs(i+1)-cPar.nrowYs(i)); // Ys
		for (i=0; i<cPar.nrowYs.n_elem-1; i++) {
			for (k=cPar.nrowYs(i); k<cPar.nrowYs(i+1); k++) {
				if (std::getline(fs, line)) {
					vec = split(line, "\t");
					Ys(i)(k-cPar.nrowYs(i)) = atof(vec[i].c_str());
					//std::cout << " vec: " << i << " "<< k << " " << vec[i];
				}
				//std::cout << cPar.ranktxt << std::endl;
			}
		}
	} else { // same overlapping samples for multiple phenotypes
		for (i=0; i<cPar.ncolYs; i++) Ys(i) = arma::zeros<arma::fvec>(cPar.nrowYs(1)); // Ys
		for (k=0; k<cPar.nrowYs(1); k++) {
			if (std::getline(fs, line)) {
				vec = split(line, "\t");
				for (i=0; i<cPar.ncolYs; i++) {
					Ys(i)(k) = atof(vec[i].c_str());
					//std::cout << " vec: " << i << " "<< k << " " << vec[i];
				}
				//std::cout << cPar.ranktxt << std::endl;
			}
		}
	}
	fs.close();
	std::cout << "Examples: Ys(0,0) = " << Ys(0)(0) << ", Ys(0,1) = " << Ys(0)(1) << cPar.ranktxt << std::endl;
	
	// Read phenotype data for testing
	if (cPar.useTest) {
		Ystest = arma::zeros<arma::fvec>(cPar.nrowYstest); // Ystest
		fs.open(szYstest.c_str(), std::fstream::in);
		for (k=0; k<cPar.nrowYstest; k++) {
			if (std::getline(fs, line)) {
				vec = split(line, "\t");
				Ystest(k) = atof(vec[0].c_str());
				//std::cout << " vec: " << " "<< k << " " << vec[i];
			}
			//std::cout << cPar.ranktxt << std::endl;
		}
		fs.close();
		std::cout << "Examples: Ystest(0) = " << Ystest(0) << ", Ystest(1) = " << Ystest(1) << cPar.ranktxt << std::endl;
	}

	std::cout << "Reading phenotype data successful !!!" << cPar.ranktxt << std::endl;
	std::cout << std::endl;
}

// Read Genotype File
void LoadGeno(arma::field<arma::fmat> & Xs, arma::fmat & Xstest, std::string szXs, std::string szXstest, PARAM & cPar) {
	int i, j, k;
	std::fstream fs;
	std::string line;
	std::vector<std::string> vec;

	std::cout << "**************************************************************" << std::endl;
	std::cout << "                   READING GENOTYPE FILES                     " << std::endl;
	std::cout << "**************************************************************" << std::endl;
	std::cout << std::endl;

	// Read genotype data for training 
	fs.open(szXs.c_str(), std::fstream::in);
	if (cPar.separinds) { // no overlapping samples for multiple phenotypes
		for (i=0; i<cPar.nrowXs.n_elem-1; i++) Xs(i) = arma::zeros<arma::fmat>(cPar.nrowXs(i+1)-cPar.nrowXs(i), cPar.ncolXs); // Xs
		for (i=0; i<cPar.nrowXs.n_elem-1; i++) {
			//std::cout << i << " i: " << cPar.nrowXs(i) << " i+1: " << cPar.nrowXs(i+1) << cPar.ranktxt << std::endl;
			for (j=cPar.nrowXs(i); j<cPar.nrowXs(i+1); j++)
				if (std::getline(fs, line)) {
					vec = split(line, "\t");
					//if (j==cPar.nrowXs(i)) std::cout << " vec: " << j << " " << vec[0] <<" " << vec[1] << cPar.ranktxt << std::endl;
					for (k=0; k<cPar.ncolXs; k++) Xs(i).row(j-cPar.nrowXs(i))(k) = (float) atof(vec[k].c_str());
				}
			//std::cout << std::endl;
		}
	} else { // same overlapping samples for multiple phenotypes
		Xs(0) = arma::zeros<arma::fmat>(cPar.nrowXs(1), cPar.ncolXs); // Xs
		//for (i=0; i<cPar.ncolYs; i++) Xs(i) = arma::zeros<arma::fmat>(cPar.nrowXs(1), cPar.ncolXs); // Xs
		for (j=0; j<cPar.nrowXs(1); j++) {
			if (std::getline(fs, line)) {
				vec = split(line, "\t");
				//if (j==cPar.nrowXs(i)) std::cout << " vec: " << j << " " << vec[0] <<" " << vec[1] << cPar.ranktxt << std::endl;
				for (k=0; k<cPar.ncolXs; k++) {
					Xs(0).row(j)(k) = (float) atof(vec[k].c_str());
					//for (i=0; i<cPar.ncolYs; i++) Xs(i).row(j)(k) = atof(vec[k].c_str());
				}
			}
		}
		for (i=1; i<cPar.ncolYs; i++) Xs(i) = Xs(0); // copy Xs(0) to other genotype matrix because all traits share the genotype
	}
	fs.close();
	for (i=0; i<cPar.ncolYs; i++) 
		std::cout << "Examples: i = " << i << ", Xs(i)(0,0) = " << Xs(i).row(0)(0) << ", Xs(i)(0,1) = " << Xs(i).row(0)(1) << cPar.ranktxt << std::endl;

	// Read genotype data for testing
	if (cPar.useTest) {
		Xstest = arma::zeros<arma::fmat>(cPar.nrowXstest, cPar.ncolXstest); // Xstest
		fs.open(szXstest.c_str(), std::fstream::in);
		for (j=0; j<cPar.nrowXstest; j++){
			if (std::getline(fs, line)) {
				vec = split(line, "\t");
				//if (j==cPar.nrowXs(i)) std::cout << " vec: " << j << " " << vec[0] <<" " << vec[1] << cPar.ranktxt << std::endl;
				for (k=0; k<cPar.ncolXstest; k++) Xstest.row(j)(k) = (float) atof(vec[k].c_str());
			}
			//std::cout << std::endl;
		}
		fs.close();
		std::cout << "Examples: Xstest(0)(0,0) = " << Xstest.row(0)(0) << ", Xstest(0)(0,1) = " << Xstest.row(0)(1) << cPar.ranktxt << std::endl;
	}
	
	std::cout << "Reading genotype data successful !!!" << cPar.ranktxt << std::endl;
	std::cout << std::endl;
}

// Read Summary Statistics
void LoadSummary(arma::fmat & b_sec, arma::fmat & seb_sec, arma::fvec & maf, std::string szSs, PARAM & cPar){
	int i, k;
	std::fstream fs;
	std::string line;
	std::vector<std::string> vec;

	std::cout << "**************************************************************" << std::endl;
	std::cout << "                   READING SUMMARY FILES                      " << std::endl;
	std::cout << "**************************************************************" << std::endl;
	std::cout << std::endl;

	maf = arma::zeros<arma::fvec>(cPar.ncolXs); // minor allele frequency
	b_sec = arma::zeros<arma::fmat>(cPar.nsecTrait, cPar.ncolXs); // beta estimates
	seb_sec = arma::zeros<arma::fmat>(cPar.nsecTrait, cPar.ncolXs); // stadard error for beta estimates

	fs.open(szSs.c_str(), std::fstream::in);
	for (i=0; i<cPar.ncolXs; i++) {
		if (std::getline(fs, line)) {
			vec = split(line, "\t");
			maf(i) = atof(vec[1].c_str());
			for (k=0; k<cPar.nsecTrait; k++) {
				b_sec(k,i) = atof(vec[k*2+2].c_str());
				seb_sec(k,i) = atof(vec[k*2+3].c_str());
			}
			//std::cout << " vec: " << i  << " " << vec[i];
		}
		//std::cout << std::endl;
	}
	fs.close();

	std::cout << "Examples: b(0,0) = " << b_sec(0,0) << ", b(0,1) = " << b_sec(0,1) << cPar.ranktxt << std::endl;
	std::cout << "Examples: seb(0,0) = " << seb_sec(0,0) << ", seb(0,1) = " << seb_sec(0,1) << cPar.ranktxt << std::endl;
	std::cout << "Reading summary data successful !!!" << cPar.ranktxt << std::endl;
	std::cout << std::endl;
}


/*************************************************************************/
/*                                                                       */
/* Functions for Coordinate Descent Algorithm                            */
/*                                                                       */
/*************************************************************************/

// Compute sgn(b)(|b|-lambda)
float _Soft(float b, float lambda) {
	float b1 = fabs(b) - lambda;
	int s = (b <= 0) ? -1 : 1;
	float b2 = (b1 <= 0) ? 0 : b1;
	return (s * b2);
}

// MCP shrinkage rule
float _MCPShrink(float bb, float dd, float lambda1, float lambda2, float gamma) {

	float t1 = 1 + lambda2 * dd;
	float t2 = gamma * lambda1 * t1;
  float res;
	if (fabs(bb) <= t2) {
		res = _Soft(bb, lambda1) * gamma / (gamma * t1 - 1);
	} else {
		res = bb / t1; 
	}
	return res;
}

// Determine the maximum lambda1
float _MaxLambda(arma::field<arma::fvec> & Ys, arma::field<arma::fmat> & Xs, 
	arma::uvec ns, int m, int p)  {
	  int i, j, n;
    float temp, zz;
  	temp = -1;
  	for (i=0; i<m; i++) {
  		n = ns(i);
  		arma::fvec rr = Ys(i);
  		for (j=0; j<p; j++) {
  			zz = arma::dot(Xs(i).col(j), rr) / n;
  			if (temp <= fabs(zz)) {
  				temp = fabs(zz);
  			}
  		}
  	}
    return (temp * 1.05);
}

// Coordinate Descent Algorithm for MCP and Laplacian Penalties
void _Cycle_M(const arma::field<arma::fvec> & Ys, const arma::field<arma::fmat> & Xs, 
	arma::fmat & b_est_v1, arma::fmat & b_sec, const arma::umat & active_set, const arma::fcube & A, 
	int m, int p, arma::uvec ns, float lambda1, float lambda2, 
	const arma::field<arma::fvec> & Xsb, bool bGWAS=FALSE, float gamma=3.0) {

	int i, j, n;
	float bij, zz, xi, bb, dd;
	arma::fvec rr;

	for (i=0; i<m; i++) {
		n = ns(i);
		if (bGWAS == FALSE) rr = Ys(i) - Xs(i) *  b_est_v1.row(i).t();
		else rr = Ys(i) - Xs(i) *  b_est_v1.row(i).t() - Xsb(i); // for GWAS data
		for (j=0; j<p; j++) {
			if (active_set(i, j) == 0)  continue;
      bij = b_est_v1(i, j);
			zz = arma::dot(Xs(i).col(j), rr) / n + bij;
			xi = lambda2 * arma::dot(b_est_v1.col(j), A.slice(j).row(i).t());
			bb = zz + xi;
			dd = arma::sum(A.slice(j).row(i));
			b_est_v1(i, j) = _MCPShrink(bb, dd, lambda1, lambda2, gamma);
			// Update residual
			rr = rr - (b_est_v1(i, j) - bij) * Xs(i).col(j);
		}
	}
}

// Coordinate Descent Algorithm for Lasso and Laplacian Penalties
void _Cycle_L(const arma::field<arma::fvec> & Ys, const arma::field<arma::fmat> & Xs, 
	arma::fmat & b_est_v1, arma::fmat & b_sec, const arma::umat & active_set, const arma::fcube & A, 
	int m, int p, arma::uvec ns, float lambda1, float lambda2, 
	const arma::field<arma::fvec> & Xsb, bool bGWAS=FALSE, float gamma=3.0) {

	int i, j, n;
	float bij, zz, xi, bb, aa;
	arma::fvec rr;

	for (i=0; i<m; i++) {
		n = ns(i);
		if (bGWAS == FALSE) rr = Ys(i) - Xs(i) *  b_est_v1.row(i).t();
		else rr = Ys(i) - Xs(i) *  b_est_v1.row(i).t() - Xsb(i); // for GWAS data
		for (j=0; j<p; j++) {
			if (active_set(i, j) == 0)  continue;
      bij = b_est_v1(i, j);
			zz = arma::dot(Xs(i).col(j), rr) / n + bij;
			xi = lambda2 * arma::dot(b_est_v1.col(j), A.slice(j).row(i).t());
			bb = zz + xi;
			aa = lambda2 * arma::sum(A.slice(j).row(i)) + 1;
			b_est_v1(i, j) = _Soft(bb, lambda1) / aa;
			// Update residual
			rr = rr - (b_est_v1(i, j) - bij) * Xs(i).col(j);
		}
	}
}


// Coordinate Descent Algorithm for MCP and Laplacian Penalties with Summary Statistics
void _Cycle_MS(const arma::field<arma::fvec> & Ys, const arma::field<arma::fmat> & Xs, 
	arma::fmat & b_est_v1, arma::fmat & b_sec, const arma::umat & active_set, const arma::fcube & A, 
	int m, int p, arma::uvec ns, float lambda1, float lambda2, 
	const arma::field<arma::fvec> & Xsb, bool bGWAS=FALSE, float gamma=3.0) {

	int i, j, k, n;
	float bij, zz, xi, bb, dd;
	arma::fvec rr, b_all;
	b_all = arma::zeros<arma::fvec>(m+b_sec.n_rows); /////////

	for (i=0; i<m; i++) {
		n = ns(i);
		if (bGWAS == FALSE) rr = Ys(i) - Xs(i) *  b_est_v1.row(i).t();
		else rr = Ys(i) - Xs(i) *  b_est_v1.row(i).t() - Xsb(i); // for GWAS data
		for (j=0; j<p; j++) {
			if (active_set(i, j) == 0)  continue;
      bij = b_est_v1(i, j);
			zz = arma::dot(Xs(i).col(j), rr) / n + bij;
			for (k=0; k<m; k++) b_all(k) = b_est_v1(k,j); /////////////////
			for (k=m; k<m+b_sec.n_rows; k++) b_all(k) = b_sec(k-m,j); /////
			xi = lambda2 * arma::dot(b_all, A.slice(j).row(i).t());
			bb = zz + xi;
			dd = arma::sum(A.slice(j).row(i));
			b_est_v1(i, j) = _MCPShrink(bb, dd, lambda1, lambda2, gamma);
			// Update residual
			rr = rr - (b_est_v1(i, j) - bij) * Xs(i).col(j);
		}
	}
}

// Coordinate Descent Algorithm for Lasso and Laplacian Penalties with Summary Statistics
void _Cycle_LS(const arma::field<arma::fvec> & Ys, const arma::field<arma::fmat> & Xs, 
	arma::fmat & b_est_v1, arma::fmat & b_sec, const arma::umat & active_set, const arma::fcube & A, 
	int m, int p, arma::uvec ns, float lambda1, float lambda2, 
	const arma::field<arma::fvec> & Xsb, bool bGWAS=FALSE, float gamma=3.0) {

	int i, j, k, n;
	float bij, zz, xi, bb, aa;
	arma::fvec rr, b_all;
	b_all = arma::zeros<arma::fvec>(m+b_sec.n_rows); /////////

	for (i=0; i<m; i++) {
		n = ns(i);
		if (bGWAS == FALSE) rr = Ys(i) - Xs(i) *  b_est_v1.row(i).t();
		else rr = Ys(i) - Xs(i) *  b_est_v1.row(i).t() - Xsb(i); // for GWAS data
		for (j=0; j<p; j++) {
			if (active_set(i, j) == 0)  continue; 
      bij = b_est_v1(i, j); 
			zz = arma::dot(Xs(i).col(j), rr) / n + bij; 
			for (k=0; k<m; k++) b_all(k) = b_est_v1(k,j); /////////////////
			for (k=m; k<m+b_sec.n_rows; k++) b_all(k) = b_sec(k-m,j); /////
			xi = lambda2 * arma::dot(b_all, A.slice(j).row(i).t());
			bb = zz + xi;
			aa = lambda2 * arma::sum(A.slice(j).row(i)) + 1;
			b_est_v1(i, j) = _Soft(bb, lambda1) / aa;
			// Update residual
			rr = rr - (b_est_v1(i, j) - bij) * Xs(i).col(j);
		}
	}
}


// Coordinate Descent Algorithm for re-estimating the coefficient
int _Re_Estimate(arma::fvec & Y, arma::fmat & X,  arma::frowvec & b_est_v1,
 	arma::urowvec & active_set, float tol=1e-4, int maxIter=10000) {
	int p1 = X.n_cols;
	int p2 = arma::sum(active_set);
	int n = Y.n_elem;
	float bij;
	b_est_v1 = arma::zeros<arma::frowvec>(p1);
	arma::fvec rr = Y;
	int iter=0;
	if (p2 == 0)  {
		return 1;
	} else {
		arma::frowvec b_est_v2 = arma::ones<arma::frowvec>(p1);
		while (arma::max(arma::abs(b_est_v1 - b_est_v2)) > tol) {
			iter += 1;
			if (iter > maxIter) {
				std::cout << "Re-estimation convergence issue!" << std::endl;
				return 0;
			}
			b_est_v2 = b_est_v1;
			for (int j=0; j<p1; j++) {
				if (active_set(j) == 0) {
					continue;
				} 
				bij = b_est_v1(j);
				b_est_v1(j) = arma::dot(X.col(j), rr) / n + bij;
				rr = rr - (b_est_v1(j) - bij) * X.col(j);
			}
		}
	}
	return 1;
}


// Evaluate the result of CV
void ComputeCVRes(arma::fvec & b_min, arma::fvec & b_min0, arma::field<arma::fmat> & mse, arma::field<arma::fmat> & b_est,	
	arma::field<arma::fvec> & lambda1, arma::fvec & lambda2, int nFold, arma::umat & nzero, arma::fvec & ctprRes, arma::fmat & cvm){
	
	int i, j, min_ind_r, min_ind_c, min_ind_r0, ind0=0;
	float cvm_min, cvm_min0;

	for (i=0; i<lambda2.n_elem; i++) if (lambda2(i)==0) ind0 = i;
	//for (i=0; i<lambda2.n_elem; i++) for (j=0; j<lambda1(i).n_elem; j++) 
	//	std::cout << "lambda2: "<< lambda2(i) << "lambda1: " << lambda1(i)(j) << std::endl;
	
	cvm = mse(0); for (i=1; i<nFold; i++) cvm = cvm + mse(i); 
	cvm = cvm/(float)nFold;

	cvm_min = cvm.min();
	cvm_min0 = cvm.col(ind0).min();

	for (i=0; i<cvm.n_rows; i++)
		for (j=0; j<cvm.n_cols; j++) 
			if (cvm(i,j)==cvm_min) {min_ind_r=i; min_ind_c=j;}

	for (i=0; i<cvm.n_rows; i++) 
		if (cvm(i,ind0)==cvm_min0) {min_ind_r0=i;}

	b_min = b_est(min_ind_c).row(min_ind_r).t();
	b_min0 = b_est(ind0).row(min_ind_r0).t();

	ctprRes(3) = lambda1(ind0)(min_ind_r0); ctprRes(4) = nzero(min_ind_r0, ind0); ctprRes(5) = cvm_min0;
	ctprRes(9) = lambda1(min_ind_c)(min_ind_r); ctprRes(10) = lambda2(min_ind_c); ctprRes(11) = nzero(min_ind_r, min_ind_c); 
	ctprRes(12) = cvm_min;

	//std::cout << "cv min0= " << cvm_min0 << " nzero=" << nzero(min_ind_r0, ind0) << std::endl;
	//std::cout << "lambda1=" << lambda1(ind0)(min_ind_r0) << " (" << min_ind_r0+1 << ")" << std::endl;

	//std::cout << "cv min= "<< cvm_min << " nzero=" << nzero(min_ind_r, min_ind_c) << std::endl;
	//std::cout << "lambda1=" << lambda1(min_ind_c)(min_ind_r) << " (" << min_ind_r+1 << ")" 
	//	<< " lambda2=" << lambda2(min_ind_c) << " (" << min_ind_c+1 << ")" << std::endl;
 
}
