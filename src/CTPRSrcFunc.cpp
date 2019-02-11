/******************************************************************************
* File: CTPRSrcFunc.cpp
* Version: CTPR 1.1
* Author: Wonil Chung, Jun Chen
* First Revised: 02.14.2014
* Last Revised:  01.29.2019
* Description: C++ Functions for Cross-Trait Penalized Regression 
******************************************************************************/


//#include <Rcpp.h>
//#include <RcppArmadillo.h>
//using namespace Rcpp;

#include <armadillo>
#include <unistd.h>
#include <string.h> 
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <algorithm>

using namespace arma;
using namespace std;

/*************************************************************************/
/*                                                                       */
/* Class for All Paramters                                              */
/*                                                                       */
/*************************************************************************/

class PARAM {
public:
	
	// For input data
	string fnameXs, fextXs, fnameXstest, fextXstest;
	string fnameSs, fextSs;
	string fnameYs, fnameYstest;
	string penalize, penalize2, output, ranktxt;
	fvec nrowXs, nrowYs; int nrowXstest, nrowYstest;
	int ncolXs, ncolYs, ncolXstest, ncolYstest;
	int useSummary, useTest, nsecTrait, separinds, nFold; 
	int nGroup, start, flamb1, llamb1, rank, size, useScaling;
	float perc;
	clock_t t_start;
	
	// For Internal use
	int useMPI, maxNz, maxIter;
	bool error, re_est, slambda2;
	float gamma, tol;
	field<fvec> lambda1;
	fvec lambda2;
	umat nzero;

};


/*************************************************************************/
/*                                                                       */
/* Functions for String                                                  */
/*                                                                       */
/*************************************************************************/

// to_string function
namespace patch {
	template < typename T > string to_string( const T& n ){
		ostringstream stm ;
		stm << n ;
		return stm.str() ;
	}
}

// Split string 
vector<string> split(string str, string sep){
	char* cstr=const_cast<char*>(str.c_str());
	char* current;
	vector<string> arr;
	current=strtok(cstr,sep.c_str());
	while(current!=NULL){
		arr.push_back(current);
		current=strtok(NULL,sep.c_str());
	}
	return arr;
}

// Trim string
void trim(string& s) {
	size_t p = s.find_first_not_of(" \t");
	s.erase(0, p);
	p = s.find_last_not_of(" \t\r");
	if (string::npos != p) s.erase(p+1);
}


/*************************************************************************/
/*                                                                       */
/* Functions for Primary Check                                           */
/*                                                                       */
/*************************************************************************/


string version = "1.1";
string date = "January 29 2019";
string year = "2017-2019";

void PrintVersion(){
	cout << "**************************************************************" << endl;
	cout << "* CTPR v" << version << " "<< date << "                       " <<endl;
	cout << "* Cross-Trait Penalized Regression                            " << endl;
	cout << "* Cross-eThnic Penalized Regression                           " << endl;
	cout << "* Developed by Wonil Chung                                    " << endl;
	cout << "**************************************************************" << endl;
	cout << endl;
	cout << "Visit https://github.com/wonilchung/CTPR for latest updates   " << endl;
	cout << "Copyright (C) " << year << " Wonil Chung                     " << endl;
	cout << "For Help, Type ./ctpr -h  or ./ctprmpi -h                    " << endl;
	cout << endl;
}

void PrintLicense(){
	PrintVersion();
	cout << "CTPR Software is freely available to only academic users.    " << endl;
	cout << endl;
}

void PrintHelp(){
	PrintVersion();
	cout << "CTPR Options: " << endl;
	cout << "  --out or --output [prefix]: specify output file prefix" << endl;
	cout << "  --dos or --dosage [filename]: specify dosage file name for training" << endl;
	cout << "  --dos-ext or --dosage-extension [ext]: specify dosage file extension for training" << endl;
	cout << "  --phe or --phenotype [filename]: specify phenotype file name for training" << endl;
	cout << "  --dos-test or --dosage-test [filename]: specify dosage file name for testing" << endl;
	cout << "  --dos-test-ext or --dosage-test-extension [ext]: specify dosage file extension for testing" << endl;
	cout << "  --phe-test or --phenotype-test [filename]: specify phenotype file name for testing" << endl;
	cout << "  --sum or --summary [filename]: specify summary file name" << endl;
	cout << "  --sum-ext or --summary-extension [ext]: specify summary file extension" << endl;
	cout << "  --num-phe or --number-phenotype [num]: specify the number of phenotypes to be analyzed (default 1)" << endl;
	cout << "  --num-sum or --number-summary [num]: specify the number of phenotypes for summary file (default 1)" << endl;
	cout << "  --separ-ind or --separate-individual [num,num,..]: specify the numbers of individuals "  << endl;
	cout << "     for each phenotypes separated by comma (,) in case of multiple phenotypes" << endl;
	cout << "  --keep [filename]: specify a list of individuals to be included in the analysis" << endl;
	cout << "  --remove [filename]: specify a list of individuals to be excluded from the analysis" << endl;
	cout << "  --include [filename]: specify a list of SNPs to be included in the analysis" << endl;
	cout << "  --exclude [filename]: specify a list of SNPs to be excluded from the analysis" << endl;
	cout << "  --scaling or --scaling-phenotype: specify for scaling secondary phenotypes " << endl;
	cout << "     using simple linear regression between phenotypes and genotypes" << endl;	
	cout << "  --penalty or --penalty-term [num]: specify the sparsity and cross-trait penalty terms " << endl; 
	cout << "     (default 1; 1: Lasso+CTPR; 2: MCP+CTPR)" << endl;
	cout << "  --nfold or --number-fold [num]: specify the number of folds for coordinate decent algorithm (default 5)" << endl;
	cout << "  --prop or --proportion [num]: specify proportion of maximum number of non-zero beta (default 0.25)" << endl;
	cout << "  --lambda2 or --lambda2-option [num]: specify value for lambda2" << endl;
	cout << "    If negative value is specified, pre-specified values are used for lambda2 (default -3)" << endl;
	cout << "    -1: lambda2=(0,0.9423,1.6028,3.3793)" << endl;
	cout << "    -2: lambda2=(0,0.0610,0.9423,0.1392,0.2425,0.3858,0.5975,0.9423,1.6028,3.3793)" << endl;
	cout << "    -3: lambda2=(0,0.0610,0.9423,0.1392,0.2425,0.3858,0.5975,0.9423,1.6028,3.3793,24.5)" << endl;
	cout << "    -4: lambda2=(0,0.0610,0.9423,0.1392,0.2425,0.3858,0.5975,0.9423,1.6028,3.3793,8.5,15.5,24.5)" << endl;
	cout << "  --ng or --number-group [num]: specify the number of group for MPI mode (default number of MPI nodes)" << endl;
	cout << "  --st or --start-number [num]: specify starting number for MPI files (default 1)" << endl;
	cout << "  --flamb1 or --first-lambda1 [num]: specify first lambda1 value for MPI mode (default 1)" << endl;
	cout << "  --llamb1 or --last-lambda1 [num]: specify last lambda1 value for MPI mode (default 100)" << endl;
	cout << endl;
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
	if (cPar.rank==0) cout << "Current Time: " << buff << ", CPU time used: " << hours << " hours, " << minutes << " minutes, " << seconds << " seconds"  << endl;
	cout << endl;
}

int AssignParameters(int argc, char **argv, PARAM &cPar){
	cout << "**************************************************************" << endl;
	cout << "                    PARAMETER ASSIGNMENT                      " << endl;
	cout << "**************************************************************" << endl;
	cout << endl;

	string str;
	vector<string> vec;
	int i=0, j=0, tmp=0;
	float tmpf = 0;

	cout << "Command Line Options:" << cPar.ranktxt << endl;
	for(i = 1; i < argc; i++) { 
		if (strcmp(argv[i], "--out")==0 || strcmp(argv[i], "--output")==0) { // output file
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.output = str;
			cout << " Output File: " << str << cPar.ranktxt << endl;
		}
		else if (strcmp(argv[i], "--dos")==0 || strcmp(argv[i], "--dosage")==0) { // dosage file
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.fnameXs = str;
			cout << " Genotype File for Training: " << str << cPar.ranktxt << endl;
		}
		else if (strcmp(argv[i], "--dos-ext")==0 || strcmp(argv[i], "--dosage-extension")==0) { // dosage extension
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.fextXs = str;
			cout << " Genotype File Extension for Training: " << str << cPar.ranktxt << endl;
		}
		else if (strcmp(argv[i], "--sum")==0 || strcmp(argv[i], "--summary")==0) { // summary statistics
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.fnameSs = str;
			cPar.useSummary = 1; ////
			cout << " Summary Data File: " << str << cPar.ranktxt << endl;
		}
		else if (strcmp(argv[i], "--sum-ext")==0 || strcmp(argv[i], "--summary-extension")==0) { // summary extension
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.fextSs = str;
			cout << " Summary Data File Extension: " << str << cPar.ranktxt << endl;
		}
		else if (strcmp(argv[i], "--phe")==0 || strcmp(argv[i], "--phenotype")==0) { // phenotype file
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.fnameYs = str;
			cout << " Phenotype File: " << str << cPar.ranktxt << endl;
		}
		else if (strcmp(argv[i], "--dos-test")==0 || strcmp(argv[i], "--dosage-test")==0) { // dosage file for testing
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.fnameXstest = str;
			cPar.useTest = 1; ////
			cout << " Genotype File for Testing: " << str << cPar.ranktxt << endl;
		}
		else if (strcmp(argv[i], "--dos-test-ext")==0 || strcmp(argv[i], "--dosage-test-extension")==0) { // dosage extension for testing
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.fextXstest = str;
			cout << " Genotype File Extention for Testing: " << str << cPar.ranktxt << endl;
		}
		else if (strcmp(argv[i], "--phe-test")==0 || strcmp(argv[i], "--phenotype-test")==0) { // phenotype file for testing
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.fnameYstest = str;
			cPar.useTest = 1; ////
			cout << " Phenotype File for Testing: " << str << cPar.ranktxt << endl;
		}
		else if (strcmp(argv[i], "--num-phe")==0 || strcmp(argv[i], "--number-phenotype")==0) { // # phenotypes to be analyzed
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.ncolYs = atoi(str.c_str());
			cout << " Number of Phenotypes to be Analyzed: " << str << cPar.ranktxt << endl;
			if (cPar.ncolYs < 1) { cPar.error = true; cout << " An error Occurred on --num-phe or --number-phenotype..." << endl; break; }
		}
		else if (strcmp(argv[i], "--num-sum")==0 || strcmp(argv[i], "--number-summary")==0) { // # phenotypes for summary statistics
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.nsecTrait = atoi(str.c_str());
			cout << " Number of Phenotypes for Summary Data to be Analyzed: " << str << cPar.ranktxt << endl;
			if (cPar.nsecTrait < 1) { cPar.error = true; cout << " An error Occurred on --num-sum or --number-summary..." << endl; break; }
		}
		else if (strcmp(argv[i], "--separ-ind")==0 || strcmp(argv[i], "--separate-individual")==0) { // vector for # individuals
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			vec = split(str, ",");  //for (j=0; j<vec.size(); j++) cout << "j" << vec[j] << cPar.ranktxt << endl;
			cPar.nrowXs.set_size(vec.size()+1); cPar.nrowXs(0)=0;
			cPar.nrowYs.set_size(vec.size()+1); cPar.nrowYs(0)=0;
			for (j=0; j<vec.size(); j++) cPar.nrowXs(j+1) = cPar.nrowXs(j) + atoi(vec[j].c_str());
			for (j=0; j<vec.size(); j++) cPar.nrowYs(j+1) = cPar.nrowYs(j) + atoi(vec[j].c_str());
			cPar.separinds = 1; // Set separinds to 1 to use separate individuals for each phenotype
			cout << " Number of Individiuals for Each Phenotype: ";
			for (j=0; j<vec.size(); j++) cout << vec[j].c_str() << " "; cout << cPar.ranktxt << endl;
			for (j=0; j<vec.size(); j++) if (atoi(vec[j].c_str())<1) { cPar.error = true; cout << " An error Occurred on --separ-ind or --separate-individual..." << endl; break; }
		}
		else if (strcmp(argv[i], "--keep")==0) { // specify a list of individuals to be included in the analysis ////////////////////
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			cout << " A list of individuals to be included..." << endl;
		}
		else if (strcmp(argv[i], "--remove")==0) { // specify a list of individuals to be excluded from the analysis /////////////////////
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			cout << " A list of individuals to be excluded..." << endl;
		}
		else if (strcmp(argv[i], "--include")==0) { // specify a list of SNPs to be included in the analysis /////////////////////
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			cout << " A list of SNPs to be included..." << endl;
		}
		else if (strcmp(argv[i], "--include")==0) { // specify a list of SNPs to be excluded in the analysis /////////////////////
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			cout << " A list of SNPs to be excluded..." << endl;
		}
		else if (strcmp(argv[i], "--scaling")==0 || strcmp(argv[i], "--scaling-phenotype")==0) { // for scaling secondary phenotypes //////////////
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			cPar.useScaling = 1;
			cout << " Conducting Linear Regression for Scaling Secondary Traits..." << endl;
		}
		else if (strcmp(argv[i], "--penalty")==0 || strcmp(argv[i], "--penalty-term")==0) { // choice for penalty terms
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			tmp = atoi(str.c_str());
			if (tmp < 1 || tmp > 2) { cPar.error = true; cout << " An error Occurred on -penalty..." << endl; break; }
			if (tmp == 1) { cPar.penalize  = "Lasso"; cPar.penalize2 = "CTPR"; } // Lasso 
			if (tmp == 2) { cPar.penalize  = "MCP"; cPar.penalize2 = "CTPR"; } // MCP 
			cout << " Penalty Terms: " << cPar.penalize << "+" << cPar.penalize2 << cPar.ranktxt << endl;
		}
		else if (strcmp(argv[i], "--nfold")==0 || strcmp(argv[i], "--number-fold")==0) { // number of folds
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.nFold = atoi(str.c_str());
			cout << " Number of Folds: " << str << cPar.ranktxt << endl;
			if (cPar.nFold < 1) { cPar.error = true; cout << " An error Occurred on --nfold or --number-fold..." << endl; break; }
		}
		else if (strcmp(argv[i], "--prop")==0 || strcmp(argv[i], "--proportion")==0) { // proportion of maximum number of non-zero beta
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.perc = atof(str.c_str());
			cout << " Proportion of Maximum Number of Non-zero Beta: " << str << cPar.ranktxt << endl;
			if (cPar.perc <= 0 || cPar.perc > 1) { cPar.error = true; cout << " An error Occurred on --prop or --proportion..." << endl; break; }
		}
		else if (strcmp(argv[i], "--lambda2")==0 || strcmp(argv[i], "--lambda2-option")==0) { // option for lambda2 vector 
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			tmpf = atof(str.c_str());
			if (tmpf < 0 && (tmpf != -1 && tmpf != -2 && tmpf != -3 && tmpf != -4)) { cPar.error = true; cout << " An error Occurred on --lambda2 or --lambda2-option..." << endl; break; }
			cPar.slambda2 = true; ////
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
			cout << " Lambda2 for cross-trait penalty: ";
			for (j=0; j<cPar.lambda2.n_elem; j++) cout << cPar.lambda2(j) << " ";
			cout << cPar.ranktxt << endl;
		}
		else if (strcmp(argv[i], "--ng")==0 || strcmp(argv[i], "--number-group")==0) { // # group for MPI
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.nGroup = atoi(str.c_str());
			cout << " Number of Group for MPI: " << str << cPar.ranktxt << endl;
			if (cPar.nGroup < 1) { cPar.error = true; cout << " An error Occurred on --ng or --num-group..." << endl; break; }
		}
		else if (strcmp(argv[i], "--st")==0 || strcmp(argv[i], "--start-number")==0) { // starting number for MPI files
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.start = atoi(str.c_str());
			cout << " Starting number for MPI files: " << str << cPar.ranktxt << endl;
			if (cPar.start < 0) { cPar.error = true; cout << " An error Occurred on --st or --start-number..." << endl; break; }
		}
		else if (strcmp(argv[i], "--flamb1")==0 || strcmp(argv[i], "--first-lambda1")==0) { // first lambda1
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.flamb1 = atoi(str.c_str());
			cout << " First lambda1: " << str << cPar.ranktxt << endl;
			if (cPar.flamb1 < 0 || cPar.flamb1 > 100) { cPar.error = true; cout << " An error Occurred on --flamb1 or --first-lambda1..." << endl; break; }
		}
		else if (strcmp(argv[i], "--llamb1")==0 || strcmp(argv[i], "--last-lambda1")==0) { // last lambda1
			if(argv[i+1] == NULL || (argv[i+1][0] == '-' && argv[i+1][1] == '-')) { continue; }
			++i;
			str.clear();
			str.assign(argv[i]);
			cPar.llamb1 = atoi(str.c_str());
			cout << " Last lambda1: " << str << cPar.ranktxt << endl;
			if (cPar.llamb1 < 0 || cPar.llamb1 > 100) { cPar.error = true; cout << " An error Occurred on --llamb1 or --last-lambda1..." << endl; break; }
		}
		else {cout << " An error Occurred on Unrecognized Option: " << argv[i] << cPar.ranktxt << endl; cPar.error = true; break;}
	}

	// Check there are any errors
	if (cPar.error) { return 0; }

	cout << endl;

	return 1;
}

int ScaleTraits(field<fvec> & Ys, field<fmat> & Xs, string szYs, PARAM &cPar){
	cout << "**************************************************************" << endl;
	cout << "                SCALING SECONDARY TRAITS                      " << endl;
	cout << "**************************************************************" << endl;
	cout << endl;
	
	int n = Ys(0).n_elem;
	int m = Ys.n_elem;
	int p = Xs(0).n_cols;

	return 1;
}

int CheckCombination(PARAM &cPar){
	cout << "**************************************************************" << endl;
	cout << "              PARAMETER COMBINATION CHECK                     " << endl;
	cout << "**************************************************************" << endl;
	cout << endl;

	if (cPar.flamb1 > cPar.llamb1) {
		cout << "Error! first lambda1 is larger than last lambda1. Please check it again..." << cPar.ranktxt << endl;
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
int CheckData(string szXs, string szXstest, string szYs, string szYstest, string szSs, PARAM & cPar){
	cout << "**************************************************************" << endl;
	cout << "       PRELIMINARY PHENOTYPE/GENOTYPE/SUMMARY FILE CHECK      " << endl;
	cout << "**************************************************************" << endl;
	cout << endl;

	// Check Genotype and Phenotype data
	fstream fs;
	int rows=0, cols=0, numItems=0;
	string val;

	cout << "Performing basic file check on training genotype data :  " << szXs << cPar.ranktxt << endl;
	cout << "Checking file..." << cPar.ranktxt << endl;
	fs.open(szXs.c_str(), fstream::in);
	if (fs.is_open()) {
		while( fs.peek() != '\n' && fs >> val ){ ++numItems; }
		cols = numItems; numItems = 0;// # of columns 
		while( fs.peek() != EOF ){ getline(fs, val); ++numItems; }
		rows = numItems; // # of rows
		cout << rows << " samples and " << cols << " markers found in the file..." << cPar.ranktxt << endl;
		cPar.ncolXs = cols; // Set # markers always ///////////////
		if (!cPar.separinds) cPar.nrowXs(1) = rows; // Set nrowXs when using overlapping samples ////////////
	} else {
		cout << "Please check the path of training genotype data : " << szXs << cPar.ranktxt << endl;
		return 0;
	}
	fs.close();
	cout << endl;

	numItems = 0;
	cout << "Performing basic file check on training phenotype data :  " << szYs << cPar.ranktxt << endl;
	cout << "Checking file..." << cPar.ranktxt << endl;
	fs.open(szYs.c_str(), fstream::in);
	if (fs.is_open()) {
		while( fs.peek() != '\n' && fs >> val ){ ++numItems; }
		cols = numItems; numItems = 0;// # of columns 
		while( fs.peek() != EOF ){ getline(fs, val); ++numItems; }
		rows = numItems; // # of rows
		cout << rows << " samples and " << cols << " phenotypes found in the file..." << cPar.ranktxt << endl;
		if (cPar.ncolYs == 0) cPar.ncolYs = cols; // Set # phenotypes only when ncolYs is not specified /////////
		if (!cPar.separinds) cPar.nrowYs(1) = rows; // Set nrowYs when using overlapping samples ////////////
	} else {
		cout << "Please check the path of training phenotype data : " << szYs << cPar.ranktxt << endl;
		return 0;
	}
	fs.close();
	cout << endl;

	if (cPar.useTest) {
		numItems = 0;
		cout << "Performing basic file check on testing genotype data :  " << szXstest << cPar.ranktxt << endl;
		cout << "Checking file..." << cPar.ranktxt << endl;
		fs.open(szXstest.c_str(), fstream::in);
		if (fs.is_open()) {
			while( fs.peek() != '\n' && fs >> val ){ ++numItems; }
			cout << endl;
			cols = numItems; numItems = 0;// # of columns 
			while( fs.peek() != EOF ){ getline(fs, val); ++numItems; }
			rows = numItems; // # of rows
			cout << rows << " samples and " << cols << " markers found in the file..." << cPar.ranktxt << endl;
			cPar.ncolXstest = cols; // Set # markers always ///////////////
			cPar.nrowXstest = rows; // Set # rows always //////////////
			
		} else {
			cout << "Please check the path of testing genotype data : " << szXstest << cPar.ranktxt << endl;
			return 0;
		}
		fs.close();
		cout << endl;
		
		numItems = 0;
		cout << "Performing basic file check on testing phenotype data :  " << szYstest << cPar.ranktxt << endl;
		cout << "Checking file..." << cPar.ranktxt << endl;
		fs.open(szYstest.c_str(), fstream::in);
		if (fs.is_open()) {
			while( fs.peek() != '\n' && fs >> val ){ ++numItems; }
			cols = numItems; numItems = 0;// # of columns 
			while( fs.peek() != EOF ){ getline(fs, val); ++numItems; }
			rows = numItems; // # of rows
			cout << rows << " samples and " << cols << " phenotypes found in the file..." << cPar.ranktxt << endl;
			cPar.ncolYstest = cols; // Set # phenotypes always /////////
			cPar.nrowYstest = rows; // Set # rows always ////////////////
		} else {
			cout << "Please check the path of testing phenotype data : " << szYstest << cPar.ranktxt << endl;
			return 0;
		}
		fs.close();
		cout << endl;
	}
	
	if (cPar.useSummary) {
		numItems = 0;
		cout << "Performing basic file check on summary data :  " << szSs << cPar.ranktxt << endl;
		cout << "Checking file..." << cPar.ranktxt << endl;
		fs.open(szSs.c_str(), fstream::in);
		if (fs.is_open()) {
			while( fs.peek() != '\n' && fs >> val ){ ++numItems; }
			cols = numItems; numItems = 0;// # of columns 
			while( fs.peek() != EOF ){ getline(fs, val); ++numItems; }
			rows = numItems; // # of rows
			if (cols < 4 || (cols-2)%2!=0) {cout << "Please double-check summary data..." << cPar.ranktxt << endl; return 0;}
			cols = (int)(cols-2)/2;
			if (cPar.nsecTrait == 0) cPar.nsecTrait = cols;
			cout << rows << " markers and " << cols << " phenotypes found in the file..." << cPar.ranktxt << endl;
		} else {
			cout << "Please check the path of summary data : " << szSs << cPar.ranktxt << endl;
			return 0;
		}
		fs.close();
		cout << endl;
	}

	cPar.maxNz = (int)cPar.ncolXs*cPar.perc; 
	cout << "Maximum Number of Non-zero beta: " << cPar.maxNz << cPar.ranktxt << endl;

	cout << "Initial basic file check on phenotype/genoytype/summary data successful !!!" << cPar.ranktxt << endl;
	cout << endl;
	
	return 1;
}

// Read Phenotype File
void LoadPheno(field<fvec> & Ys, fvec & Ystest, string szYs, string szYstest, PARAM & cPar) {
	int i, k;
	fstream fs;
	string line;
	vector<string> vec;

	cout << "**************************************************************" << endl;
	cout << "                   READING PHENOTYPE FILES                    " << endl;
	cout << "**************************************************************" << endl;
	cout << endl;
	
	// Read phenotype data for training 
	fs.open(szYs.c_str(), fstream::in);
	if (cPar.separinds) { // no overlapping samples for multiple phenotypes
		for (i=0; i<cPar.nrowYs.n_elem-1; i++) Ys(i) = zeros<fvec>(cPar.nrowYs(i+1)-cPar.nrowYs(i)); // Ys
		for (i=0; i<cPar.nrowYs.n_elem-1; i++) {
			for (k=cPar.nrowYs(i); k<cPar.nrowYs(i+1); k++) {
				if (getline(fs, line)) {
					vec = split(line, "\t");
					Ys(i)(k-cPar.nrowYs(i)) = atof(vec[i].c_str());
					//cout << " vec: " << i << " "<< k << " " << vec[i];
				}
				//cout << cPar.ranktxt << endl;
			}
		}
	} else { // same overlapping samples for multiple phenotypes
		for (i=0; i<cPar.ncolYs; i++) Ys(i) = zeros<fvec>(cPar.nrowYs(1)); // Ys
		for (k=0; k<cPar.nrowYs(1); k++) {
			if (getline(fs, line)) {
				vec = split(line, "\t");
				for (i=0; i<cPar.ncolYs; i++) {
					Ys(i)(k) = atof(vec[i].c_str());
					//cout << " vec: " << i << " "<< k << " " << vec[i];
				}
				//cout << cPar.ranktxt << endl;
			}
		}
	}
	fs.close();
	cout << "Examples: Ys(0,0) = " << Ys(0)(0) << ", Ys(0,1) = " << Ys(0)(1) << cPar.ranktxt << endl;
	
	// Read phenotype data for testing
	if (cPar.useTest) {
		Ystest = zeros<fvec>(cPar.nrowYstest); // Ystest
		fs.open(szYstest.c_str(), fstream::in);
		for (k=0; k<cPar.nrowYstest; k++) {
			if (getline(fs, line)) {
				vec = split(line, "\t");
				Ystest(k) = atof(vec[0].c_str());
				//cout << " vec: " << " "<< k << " " << vec[i];
			}
			//cout << cPar.ranktxt << endl;
		}
		fs.close();
		cout << "Examples: Ystest(0) = " << Ystest(0) << ", Ystest(1) = " << Ystest(1) << cPar.ranktxt << endl;
	}

	cout << "Reading phenotype data successful !!!" << cPar.ranktxt << endl;
	cout << endl;
}

// Read Genotype File
void LoadGeno(field<fmat> & Xs, fmat & Xstest, string szXs, string szXstest, PARAM & cPar) {
	int i, j, k;
	fstream fs;
	string line;
	vector<string> vec;

	cout << "**************************************************************" << endl;
	cout << "                   READING GENOTYPE FILES                     " << endl;
	cout << "**************************************************************" << endl;
	cout << endl;

	// Read genotype data for training 
	fs.open(szXs.c_str(), fstream::in);
	if (cPar.separinds) { // no overlapping samples for multiple phenotypes
		for (i=0; i<cPar.nrowXs.n_elem-1; i++) Xs(i) = zeros<fmat>(cPar.nrowXs(i+1)-cPar.nrowXs(i), cPar.ncolXs); // Xs
		for (i=0; i<cPar.nrowXs.n_elem-1; i++) {
			//cout << i << " i: " << cPar.nrowXs(i) << " i+1: " << cPar.nrowXs(i+1) << cPar.ranktxt << endl;
			for (j=cPar.nrowXs(i); j<cPar.nrowXs(i+1); j++)
				if (getline(fs, line)) {
					vec = split(line, "\t");
					//if (j==cPar.nrowXs(i)) cout << " vec: " << j << " " << vec[0] <<" " << vec[1] << cPar.ranktxt << endl;
					for (k=0; k<cPar.ncolXs; k++) Xs(i).row(j-cPar.nrowXs(i))(k) = (float) atof(vec[k].c_str());
				}
			//cout << endl;
		}
	} else { // same overlapping samples for multiple phenotypes
		Xs(0) = zeros<fmat>(cPar.nrowXs(1), cPar.ncolXs); // Xs
		//for (i=0; i<cPar.ncolYs; i++) Xs(i) = zeros<fmat>(cPar.nrowXs(1), cPar.ncolXs); // Xs
		for (j=0; j<cPar.nrowXs(1); j++) {
			if (getline(fs, line)) {
				vec = split(line, "\t");
				//if (j==cPar.nrowXs(i)) cout << " vec: " << j << " " << vec[0] <<" " << vec[1] << cPar.ranktxt << endl;
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
		cout << "Examples: i = " << i << ", Xs(i)(0,0) = " << Xs(i).row(0)(0) << ", Xs(i)(0,1) = " << Xs(i).row(0)(1) << cPar.ranktxt << endl;

	// Read genotype data for testing
	if (cPar.useTest) {
		Xstest = zeros<fmat>(cPar.nrowXstest, cPar.ncolXstest); // Xstest
		fs.open(szXstest.c_str(), fstream::in);
		for (j=0; j<cPar.nrowXstest; j++){
			if (getline(fs, line)) {
				vec = split(line, "\t");
				//if (j==cPar.nrowXs(i)) cout << " vec: " << j << " " << vec[0] <<" " << vec[1] << cPar.ranktxt << endl;
				for (k=0; k<cPar.ncolXstest; k++) Xstest.row(j)(k) = (float) atof(vec[k].c_str());
			}
			//cout << endl;
		}
		fs.close();
		cout << "Examples: Xstest(0)(0,0) = " << Xstest.row(0)(0) << ", Xstest(0)(0,1) = " << Xstest.row(0)(1) << cPar.ranktxt << endl;
	}
	
	cout << "Reading genotype data successful !!!" << cPar.ranktxt << endl;
	cout << endl;
}

// Read Summary Statistics
void LoadSummary(fmat & b_sec, fmat & seb_sec, fvec & maf, string szSs, PARAM & cPar){
	int i, k;
	fstream fs;
	string line;
	vector<string> vec;

	cout << "**************************************************************" << endl;
	cout << "                   READING SUMMARY FILES                      " << endl;
	cout << "**************************************************************" << endl;
	cout << endl;

	maf = zeros<fvec>(cPar.ncolXs); // minor allele frequency
	b_sec = zeros<fmat>(cPar.nsecTrait, cPar.ncolXs); // beta estimates
	seb_sec = zeros<fmat>(cPar.nsecTrait, cPar.ncolXs); // stadard error for beta estimates

	fs.open(szSs.c_str(), fstream::in);
	for (i=0; i<cPar.ncolXs; i++) {
		if (getline(fs, line)) {
			vec = split(line, "\t");
			maf(i) = atof(vec[1].c_str());
			for (k=0; k<cPar.nsecTrait; k++) {
				b_sec(k,i) = atof(vec[k*2+2].c_str());
				seb_sec(k,i) = atof(vec[k*2+3].c_str());
			}
			//cout << " vec: " << i  << " " << vec[i];
		}
		//cout << endl;
	}
	fs.close();

	cout << "Examples: b(0,0) = " << b_sec(0,0) << ", b(0,1) = " << b_sec(0,1) << cPar.ranktxt << endl;
	cout << "Examples: seb(0,0) = " << seb_sec(0,0) << ", seb(0,1) = " << seb_sec(0,1) << cPar.ranktxt << endl;
	cout << "Reading summary data successful !!!" << cPar.ranktxt << endl;
	cout << endl;
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
float _MaxLambda(field<fvec> & Ys, field<fmat> & Xs, 
	uvec ns, int m, int p)  {
	  int i, j, n;
    float temp, zz;
  	temp = -1;
  	for (i=0; i<m; i++) {
  		n = ns(i);
  		fvec rr = Ys(i);
  		for (j=0; j<p; j++) {
  			zz = dot(Xs(i).col(j), rr) / n;
  			if (temp <= fabs(zz)) {
  				temp = fabs(zz);
  			}
  		}
  	}
    return (temp * 1.05);
}

// Coordinate Descent Algorithm for MCP and Laplacian Penalties
void _Cycle_M(const field<fvec> & Ys, const field<fmat> & Xs, 
	fmat & b_est_v1, fmat & b_sec, const umat & active_set, const fcube & A, 
	int m, int p, uvec ns, float lambda1, float lambda2, 
	const field<fvec> & Xsb, bool bGWAS=false, float gamma=3.0) {

	int i, j, n;
	float bij, zz, xi, bb, dd;
	fvec rr;

	for (i=0; i<m; i++) {
		n = ns(i);
		if (bGWAS == false) rr = Ys(i) - Xs(i) *  b_est_v1.row(i).t();
		else rr = Ys(i) - Xs(i) *  b_est_v1.row(i).t() - Xsb(i); // for GWAS data
		for (j=0; j<p; j++) {
			if (active_set(i, j) == 0)  continue;
      bij = b_est_v1(i, j);
			zz = dot(Xs(i).col(j), rr) / n + bij;
			xi = lambda2 * dot(b_est_v1.col(j), A.slice(j).row(i).t());
			bb = zz + xi;
			dd = sum(A.slice(j).row(i));
			b_est_v1(i, j) = _MCPShrink(bb, dd, lambda1, lambda2, gamma);
			// Update residual
			rr = rr - (b_est_v1(i, j) - bij) * Xs(i).col(j);
		}
	}
}

// Coordinate Descent Algorithm for Lasso and Laplacian Penalties
void _Cycle_L(const field<fvec> & Ys, const field<fmat> & Xs, 
	fmat & b_est_v1, fmat & b_sec, const umat & active_set, const fcube & A, 
	int m, int p, uvec ns, float lambda1, float lambda2, 
	const field<fvec> & Xsb, bool bGWAS=false, float gamma=3.0) {

	int i, j, n;
	float bij, zz, xi, bb, aa;
	fvec rr;

	for (i=0; i<m; i++) {
		n = ns(i);
		if (bGWAS == false) rr = Ys(i) - Xs(i) *  b_est_v1.row(i).t();
		else rr = Ys(i) - Xs(i) *  b_est_v1.row(i).t() - Xsb(i); // for GWAS data
		for (j=0; j<p; j++) {
			if (active_set(i, j) == 0)  continue;
      bij = b_est_v1(i, j);
			zz = dot(Xs(i).col(j), rr) / n + bij;
			xi = lambda2 * dot(b_est_v1.col(j), A.slice(j).row(i).t());
			bb = zz + xi;
			aa = lambda2 * sum(A.slice(j).row(i)) + 1;
			b_est_v1(i, j) = _Soft(bb, lambda1) / aa;
			// Update residual
			rr = rr - (b_est_v1(i, j) - bij) * Xs(i).col(j);
		}
	}
}


// Coordinate Descent Algorithm for MCP and Laplacian Penalties with Summary Statistics
void _Cycle_MS(const field<fvec> & Ys, const field<fmat> & Xs, 
	fmat & b_est_v1, fmat & b_sec, const umat & active_set, const fcube & A, 
	int m, int p, uvec ns, float lambda1, float lambda2, 
	const field<fvec> & Xsb, bool bGWAS=false, float gamma=3.0) {

	int i, j, k, n;
	float bij, zz, xi, bb, dd;
	fvec rr, b_all;
	b_all = zeros<fvec>(m+b_sec.n_rows); /////////

	for (i=0; i<m; i++) {
		n = ns(i);
		if (bGWAS == false) rr = Ys(i) - Xs(i) *  b_est_v1.row(i).t();
		else rr = Ys(i) - Xs(i) *  b_est_v1.row(i).t() - Xsb(i); // for GWAS data
		for (j=0; j<p; j++) {
			if (active_set(i, j) == 0)  continue;
      bij = b_est_v1(i, j);
			zz = dot(Xs(i).col(j), rr) / n + bij;
			for (k=0; k<m; k++) b_all(k) = b_est_v1(k,j); /////////////////
			for (k=m; k<m+b_sec.n_rows; k++) b_all(k) = b_sec(k-m,j); /////
			xi = lambda2 * dot(b_all, A.slice(j).row(i).t());
			bb = zz + xi;
			dd = sum(A.slice(j).row(i));
			b_est_v1(i, j) = _MCPShrink(bb, dd, lambda1, lambda2, gamma);
			// Update residual
			rr = rr - (b_est_v1(i, j) - bij) * Xs(i).col(j);
		}
	}
}

// Coordinate Descent Algorithm for Lasso and Laplacian Penalties with Summary Statistics
void _Cycle_LS(const field<fvec> & Ys, const field<fmat> & Xs, 
	fmat & b_est_v1, fmat & b_sec, const umat & active_set, const fcube & A, 
	int m, int p, uvec ns, float lambda1, float lambda2, 
	const field<fvec> & Xsb, bool bGWAS=false, float gamma=3.0) {

	int i, j, k, n;
	float bij, zz, xi, bb, aa;
	fvec rr, b_all;
	b_all = zeros<fvec>(m+b_sec.n_rows); /////////

	for (i=0; i<m; i++) {
		n = ns(i);
		if (bGWAS == false) rr = Ys(i) - Xs(i) *  b_est_v1.row(i).t();
		else rr = Ys(i) - Xs(i) *  b_est_v1.row(i).t() - Xsb(i); // for GWAS data
		for (j=0; j<p; j++) {
			if (active_set(i, j) == 0)  continue; 
      bij = b_est_v1(i, j); 
			zz = dot(Xs(i).col(j), rr) / n + bij; 
			for (k=0; k<m; k++) b_all(k) = b_est_v1(k,j); /////////////////
			for (k=m; k<m+b_sec.n_rows; k++) b_all(k) = b_sec(k-m,j); /////
			xi = lambda2 * dot(b_all, A.slice(j).row(i).t());
			bb = zz + xi;
			aa = lambda2 * sum(A.slice(j).row(i)) + 1;
			b_est_v1(i, j) = _Soft(bb, lambda1) / aa;
			// Update residual
			rr = rr - (b_est_v1(i, j) - bij) * Xs(i).col(j);
		}
	}
}


// Coordinate Descent Algorithm for re-estimating the coefficient
int _Re_Estimate(fvec & Y, fmat & X,  frowvec & b_est_v1,
 	urowvec & active_set, float tol=1e-4, int maxIter=10000) {
	int p1 = X.n_cols;
	int p2 = sum(active_set);
	int n = Y.n_elem;
	float bij;
	b_est_v1 = zeros<frowvec>(p1);
	fvec rr = Y;
	int iter=0;
	if (p2 == 0)  {
		return 1;
	} else {
		frowvec b_est_v2 = ones<frowvec>(p1);
		while (max(abs(b_est_v1 - b_est_v2)) > tol) {
			iter += 1;
			if (iter > maxIter) {
				cout << "Re-estimation convergence issue!" << endl;
				return 0;
			}
			b_est_v2 = b_est_v1;
			for (int j=0; j<p1; j++) {
				if (active_set(j) == 0) {
					continue;
				} 
				bij = b_est_v1(j);
				b_est_v1(j) = dot(X.col(j), rr) / n + bij;
				rr = rr - (b_est_v1(j) - bij) * X.col(j);
			}
		}
	}
	return 1;
}


// Evaluate the result of CV
void ComputeCVRes(fvec & b_min, fvec & b_min0, field<fmat> & mse, field<fmat> & b_est,	
	field<fvec> & lambda1, fvec & lambda2, int nFold, umat & nzero, fvec & ctprRes, fmat & cvm){
	
	int i, j, min_ind_r, min_ind_c, min_ind_r0, ind0=0;
	float cvm_min, cvm_min0;

	for (i=0; i<lambda2.n_elem; i++) if (lambda2(i)==0) ind0 = i;
	//for (i=0; i<lambda2.n_elem; i++) for (j=0; j<lambda1(i).n_elem; j++) 
	//	cout << "lambda2: "<< lambda2(i) << "lambda1: " << lambda1(i)(j) << endl;
	
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

	//cout << "cv min0= " << cvm_min0 << " nzero=" << nzero(min_ind_r0, ind0) << endl;
	//cout << "lambda1=" << lambda1(ind0)(min_ind_r0) << " (" << min_ind_r0+1 << ")" << endl;

	//cout << "cv min= "<< cvm_min << " nzero=" << nzero(min_ind_r, min_ind_c) << endl;
	//cout << "lambda1=" << lambda1(min_ind_c)(min_ind_r) << " (" << min_ind_r+1 << ")" 
	//	<< " lambda2=" << lambda2(min_ind_c) << " (" << min_ind_c+1 << ")" << endl;
 
}
