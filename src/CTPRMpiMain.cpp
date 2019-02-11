/******************************************************************************
* File: CTPRMpiMain.cpp
* Version: CTPR 1.1
* Author: Wonil Chung
* First Revised: 02.14.2014
* Last Revised:  01.29.2019
* Description: C++ Code for Cross-Trait Penalized Regression  
*  for Message Passing Interface (MPI) Program
******************************************************************************/


//#include <Rcpp.h>
//#include <RcppArmadillo.h>
//using namespace Rcpp; 

#include <mpi.h>
#include <armadillo>
#include "CTPRSrcFunc.cpp"

using namespace arma;
using namespace std;

/*************************************************************************/
/*                                                                       */
/* MPI Code for CTPR using C code                                        */
/*                                                                       */
/*************************************************************************/


// Function for Coordinate Decent Algorithm 
void CTPRMpiProc(field<fvec> & Ys, field<fmat> & Xs, const fcube & A, field<fvec> & lambda1, fvec lambda2,
	field<fmat> & b_est, field<fmat> & b_est_2, field<fmat> & b_re_est, fmat & b_sec, umat & nzero, PARAM & cPar) {
	
	// Determine a sparcity penaty term (Lasso vs MCP) 
	void (* _Cycle) (const field<fvec> &, const field<fmat> &,
		fmat &, fmat &, const umat &, const fcube &, int, int, uvec, float, float, const field<fvec> &, bool, float);

	if (cPar.penalize == "MCP" && cPar.penalize2 == "CTPR" && cPar.useSummary == 0) {
		_Cycle = & _Cycle_M;
		cout << "MCP and CTPR penalties are used." << endl;
	} else if (cPar.penalize == "Lasso" && cPar.penalize2 == "CTPR" && cPar.useSummary == 0){
		_Cycle = & _Cycle_L;
		cout << "Lasso and CTPR penalties are used." << endl;
	} else if (cPar.penalize == "MCP" && cPar.penalize2 == "CTPR" && cPar.useSummary == 1) {
		_Cycle = & _Cycle_MS;
		cout << "MCP and CTPR penalties are used with Summary Statistics." << endl;
	} else if (cPar.penalize == "Lasso" && cPar.penalize2 == "CTPR" && cPar.useSummary == 1){
		_Cycle = & _Cycle_LS;
		cout << "Lasso and CTPR penalties are used with Summary Statistics." << endl;
	} 

	int maxNz = cPar.maxNz;
	bool re_est = cPar.re_est;
	float tol = cPar.tol; 
	float gamma = cPar.gamma; 
	int maxIter = cPar.maxIter; 
	int flamb1 = cPar.flamb1; 
	int llamb1 = cPar.llamb1; 
	int nGroup = cPar.nGroup;
	int rank = cPar.rank; 
	int size = cPar.size; 

	int i, j;
	int m = Ys.n_elem;
	int p = Xs(0).n_cols;
	int q = lambda2.n_elem;
	uvec ns(m);
	b_est.set_size(q);
	b_est_2.set_size(q);

	// Retrieve the centers and Center X and Y
	fmat Xs_c(m, p);
	fmat Xs_s(m, p);
	fvec Ys_c(m);

	//if (rank==0) {cout << "ori y=" << " " << Ys(0)(0) << " " << Ys(0)(1) << Ys(1)(0) << " " << Ys(1)(1) << endl;}
	//if (rank==0) {cout << "ori x=" << " " << Xs(0)(0,0) << " " << Xs(0)(0,1) << Xs(0)(0,2) << " " << Xs(0)(0,3) << endl;}

	for (i=0; i<m; i++) {
		ns(i) = Xs(i).n_rows;
		Xs_c.row(i) = mean(Xs(i));
		Xs_s.row(i) = stddev(Xs(i), 1);
		Xs(i).each_row() -= Xs_c.row(i); 
		Xs(i).each_row() /= Xs_s.row(i);
		Ys_c(i) = mean(Ys(i));
		Ys(i) -= Ys_c(i);
	}

	//if (rank==0) {cout << "cen y=" << " " << Ys(0)(0) << " " << Ys(0)(1) << Ys(1)(0) << " " << Ys(1)(1) << endl;}
	//if (rank==0) {cout << "cen x=" << " " << Xs(0)(0,0) << " " << Xs(0)(0,1) << Xs(0)(0,2) << " " << Xs(0)(0,3) << endl;}

	// Determine the maximum lambda1
	if (lambda1.n_elem == 0) {
		cout << "Determining maximum lambda1 ..." << endl; 
		float _lambda1_max, lambda1_max; 
		_lambda1_max = _MaxLambda(Ys, Xs, ns, 1, p); lambda1_max = 0;
		float *buffer = new float[size];
		MPI_Allgather(&_lambda1_max, 1, MPI_FLOAT, buffer, 1, MPI_FLOAT, MPI_COMM_WORLD); //////////
		for (i=0; i<size; i++) if (lambda1_max <= buffer[i]) lambda1_max = buffer[i];
		delete[] buffer;
		lambda1.set_size(q);
		for (i=0; i<q; i++) {
			//lambda1(i) = lambda1_max * exp(-6.907755 * linspace<fvec>(0, 1, 101));  /////////////////////////////////////////
			lambda1(i) = lambda1_max * exp(-6.907755 * linspace<fvec>((float)flamb1/100.0, (float)llamb1/100.0, llamb1-flamb1+1));  
		}
	} else {
		if (lambda1.n_elem == 1 && q > 1) {
			fvec tempt = lambda1(0);
			lambda1.set_size(q);
			for (i=0; i<q; i++) {
				lambda1(i) = tempt;
			}
		} else {
			if (lambda1.n_elem != q) {
				cout << "The list of lambda1 should have the same length as lambda2..." << endl;
				return;
			}
		}
	}

	int iter, flag = 0;
	float _lambda1, _lambda2;
	float xbarb;
	fmat _b_est_v1(m, p);
	fmat _b_est_v2(m, p); 
	fmat _b_est_st(m, p);
  umat active_set(m, p);
  umat active_set2(m, p);
  frowvec temp2(p+1);

  int L1 = 0;
  int L2 = q;
  for (i=0; i<lambda1.n_elem; i++) {
  	L1 = lambda1(i).n_elem > L1 ? lambda1(i).n_elem : L1;
  }
  fmat RSS(L1, L2);
  //RSS.fill(NA_REAL);
	RSS.fill(datum::nan);

	// Set Parameters for MPI Code ////////////////////////////////////////////////////////////////////////////////////
	int k, w, e, z, nXsb=0, cnt=0, eLoop=0;
	bool bGWAS = true;
	
	field<fvec> Xsb(m);
	for (w=0; w<m; w++) Xsb(w) = zeros<fvec>(Ys(w).n_elem); 
	for (w=0; w<m; w++) nXsb += Ys(w).n_elem;

	int *alleLoop = new int[size];
	float *curXsb = new float[nXsb]; 
	float *sumXsb = new float[nXsb];
	float *allXsb = new float[nXsb*size];
	float *allxbarb = new float[size]; 
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	///////////////////////////////////////
	//fmat objval(10, L1); // Save object function
	///////////////////////////////////////

	//if (size % nGroup !=0) {
	//	cout << "Please re-set the number of group for parallel computing!" << endl;
	//	return;
	//}

	// Conduct Coordinate Descent Algorithm
  for (j=0; j<q; j++) { // for j
  	cout << "*"; cout << lambda1(j).n_elem;
		_lambda2 = lambda2(j); // Set lambda2

  	fmat _b_est(lambda1(j).n_elem, p+1);
		fmat _b_est_2(lambda1(j).n_elem, p+1); /////////////////

	/////////////////////////////////////
	//objval.fill(0);
	/////////////////////////////////////
 
  	for (i=0; i<lambda1(j).n_elem; i++) { // for i
			cout << "~";
			_lambda1 = lambda1(j)(i); // Set lambda1

      if (i == 0) { _b_est_v1.zeros(); } // Basically, initial values for all beta are zero	
      else { _b_est_v1 = _b_est_st; }

			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			// Receive Xsb from other jobs and run one complete cycle (1)
			if (i == 0) {
				for (w=0; w<nXsb; w++) curXsb[w] = 0;
				for (w=0; w<m; w++) Xsb(w) = zeros<fvec>(Ys(w).n_elem); 
			} else {
				MPI_Allgather(curXsb, nXsb, MPI_FLOAT, allXsb, nXsb, MPI_FLOAT, MPI_COMM_WORLD); //////////////
				for (w=0; w<nXsb; w++) sumXsb[w] = 0;
				for (k=0; k<size; k++){ if (k==rank) continue; for (w=0; w<nXsb; w++) sumXsb[w] += allXsb[k*nXsb+w]; }
				cnt=0; for (w=0; w<m; w++) for (k=0; k<Ys(w).n_elem; k++) {Xsb(w)(k) = sumXsb[cnt]; cnt++;}
			}
			
			for (e=0; e<nGroup; e++) 
			if (rank % nGroup == e) {
				_Cycle(Ys, Xs, _b_est_v1, b_sec, ones<umat>(m, p), A, m, p, ns, _lambda1, _lambda2, Xsb, bGWAS, gamma); 
				for (w=0; w<m; w++) {Xsb(w) = Xs(w) * _b_est_v1.row(w).t();}
				cnt=0; for (w=0; w<m; w++) for (k=0; k<Ys(w).n_elem; k++) {curXsb[cnt] = Xsb(w)(k); cnt++;}
			} else {
				MPI_Allgather(curXsb, nXsb, MPI_FLOAT, allXsb, nXsb, MPI_FLOAT, MPI_COMM_WORLD); //////////////
				for (w=0; w<nXsb; w++) sumXsb[w] = 0;
				for (k=0; k<size; k++){ if (k==rank) continue; for (w=0; w<nXsb; w++) sumXsb[w] += allXsb[k*nXsb+w]; }
				cnt=0; for (w=0; w<m; w++) for (k=0; k<Ys(w).n_elem; k++) {Xsb(w)(k) = sumXsb[cnt]; cnt++;}
			}
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			// Iterate on only the active set till convergence
      iter = 0; 
      while (true) { // while 1
      	active_set = (abs(_b_est_v1) > datum::eps); 
      	_b_est_v2.ones(); 
      	while (true)  { // while 2
      		iter += 1;
					//cout << rank << " " << iter << endl;
					// Exit the loop (1) /////////////////////////////////////////////////////////////////////////////////////////////
					eLoop = 0;
					if (max(max(abs(_b_est_v1 - _b_est_v2))) <= tol) eLoop = 1;
					MPI_Allgather(&eLoop, 1, MPI_INT, alleLoop, 1, MPI_INT, MPI_COMM_WORLD); ////////////////
					cnt = 0; for (w=0; w<size; w++) cnt += alleLoop[w];
					if (cnt==size) break;
					////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
				
      		_b_est_v2 = _b_est_v1;
      		if (iter >= maxIter) {
      			cout << "Exceeds the maximum iteration for lambda:" << _lambda1 <<"," << _lambda2;
      			cout << ". Further lambda1 will not be considered."; // << endl;
      			flag = 1; 
      			break; 
      		}

					///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
					// Receive Xsb from other jobs and run one complete cycle (2)
					MPI_Allgather(curXsb, nXsb, MPI_FLOAT, allXsb, nXsb, MPI_FLOAT, MPI_COMM_WORLD); /////////////////
					for (w=0; w<nXsb; w++) sumXsb[w] = 0;
					for (k=0; k<size; k++){ if (k==rank) continue; for (w=0; w<nXsb; w++) sumXsb[w] += allXsb[k*nXsb+w]; }
					cnt=0; for (w=0; w<m; w++) for (k=0; k<Ys(w).n_elem; k++) {Xsb(w)(k) = sumXsb[cnt]; cnt++;}	

					for (e=0; e<nGroup; e++) 
					if (rank % nGroup == e) {
						_Cycle(Ys, Xs, _b_est_v1, b_sec, active_set, A, m, p, ns, _lambda1, _lambda2, Xsb, bGWAS, gamma);
						for (w=0; w<m; w++) {Xsb(w) = Xs(w) * _b_est_v1.row(w).t();}
						cnt=0; for (w=0; w<m; w++) for (k=0; k<Ys(w).n_elem; k++) {curXsb[cnt] = Xsb(w)(k); cnt++;}	
					} else {
						MPI_Allgather(curXsb, nXsb, MPI_FLOAT, allXsb, nXsb, MPI_FLOAT, MPI_COMM_WORLD); /////////////////
						for (w=0; w<nXsb; w++) sumXsb[w] = 0;
						for (k=0; k<size; k++){ if (k==rank) continue; for (w=0; w<nXsb; w++) sumXsb[w] += allXsb[k*nXsb+w]; }
						cnt=0; for (w=0; w<m; w++) for (k=0; k<Ys(w).n_elem; k++) {Xsb(w)(k) = sumXsb[cnt]; cnt++;}	
					}
					///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      	} // end while 2

      	if (flag == 1) break;
				
				///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
				// Receive Xsb from other jobs and run one complete cycle (3)
				MPI_Allgather(curXsb, nXsb, MPI_FLOAT, allXsb, nXsb, MPI_FLOAT, MPI_COMM_WORLD); //////////////////
				for (w=0; w<nXsb; w++) sumXsb[w] = 0;
				for (k=0; k<size; k++){ if (k==rank) continue; for (w=0; w<nXsb; w++) sumXsb[w] += allXsb[k*nXsb+w]; }
				cnt=0; for (w=0; w<m; w++) for (k=0; k<Ys(w).n_elem; k++) {Xsb(w)(k) = sumXsb[cnt]; cnt++;}	

				for (e=0; e<nGroup; e++) 
				if (rank % nGroup == e) {
					_Cycle(Ys, Xs, _b_est_v1, b_sec, ones<umat>(m, p), A, m, p, ns, _lambda1, _lambda2, Xsb, bGWAS, gamma);
					for (w=0; w<m; w++) {Xsb(w) = Xs(w) * _b_est_v1.row(w).t();}
					cnt=0; for (w=0; w<m; w++) for (k=0; k<Ys(w).n_elem; k++) {curXsb[cnt] = Xsb(w)(k); cnt++;}	
				} else {
					MPI_Allgather(curXsb, nXsb, MPI_FLOAT, allXsb, nXsb, MPI_FLOAT, MPI_COMM_WORLD); /////////////////
					for (w=0; w<nXsb; w++) sumXsb[w] = 0;
					for (k=0; k<size; k++){ if (k==rank) continue; for (w=0; w<nXsb; w++) sumXsb[w] += allXsb[k*nXsb+w]; }
					cnt=0; for (w=0; w<m; w++) for (k=0; k<Ys(w).n_elem; k++) {Xsb(w)(k) = sumXsb[cnt]; cnt++;}	
				}
				///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

      	active_set2 = (abs(_b_est_v1) > datum::eps);

				// Exit the loop (2) /////////////////////////////////////////////////////////////////////////////////////////////
				eLoop = 0;
      	if (accu(active_set2 != active_set) < 0.5) eLoop = 1;
				MPI_Allgather(&eLoop, 1, MPI_INT, alleLoop, 1, MPI_INT, MPI_COMM_WORLD); ///////////////////
				cnt = 0; for (w=0; w<size; w++) cnt += alleLoop[w];
				if (cnt==size) break;
				////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      } // end while 1

			// Save object function /////////////////////////////////////
			/*MPI_Allgather(curXsb, nXsb, MPI_FLOAT, allXsb, nXsb, MPI_FLOAT, MPI_COMM_WORLD); // MPI_Allgather
			for (w=0; w<nXsb; w++) sumXsb[w] = 0;
			for (k=0; k<size; k++) for (w=0; w<nXsb; w++) sumXsb[w] += allXsb[k*nXsb+w];
			cnt=0; for (w=0; w<m; w++) for (k=0; k<Ys(w).n_elem; k++) {Xsb(w)(k) = sumXsb[cnt]; cnt++;}	 
			fvec rr; float zz=0, sp=0, ct=0, obj=0; float *abc = new float[size]; for (w=0; w<size; w++) abc[w]=0;
			for (w=0; w<m; w++) { rr = Ys(w) - Xsb(w); zz += dot(rr, rr)/(2.0*ns(w)); sp += _lambda1*sum(abs(_b_est_v1.row(w).t())); }
			rr = _b_est_v1.row(0).t() - _b_est_v1.row(1).t(); ct = (_lambda2/2.0)*dot(rr, rr); obj = sp + ct;
			MPI_Allgather(&obj, 1, MPI_FLOAT, abc, 1, MPI_FLOAT, MPI_COMM_WORLD); // MPI_Allgather
			obj=0; for (w=0; w<size; w++) obj += abc[w]; obj += zz;
			objval(0,i)=_lambda2; objval(1,i)=_lambda1; objval(2,i)=iter; objval(3,i)=Ys(0)(0); objval(4,i)=Xsb(0)(0); objval(5,i)=_b_est_v1(0,0); 
			objval(6,i)= sum(abs(_b_est_v1.row(0)) > datum::eps); objval(7,i)=sum(_b_est_v1.row(0).t()); 
			objval(8,i)= max(abs(_b_est_v1.row(0))); objval(9,i)=obj; delete[] abc;*/
			/////////////////////////////////////////////////////////////

      if (flag == 1) { flag = 0; break; } 

			// Exit the loop (3) /////////////////////////////////////////////////////////////////////////////////////////////
			eLoop = 0;
      if (sum(abs(_b_est_v1.row(0)) > datum::eps) > maxNz) eLoop = 1;
			MPI_Allgather(&eLoop, 1, MPI_INT, alleLoop, 1, MPI_INT, MPI_COMM_WORLD); //////////////////
			cnt = 0; for (w=0; w<size; w++) cnt += alleLoop[w];
			if (cnt==size) break;
			////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

      // Convert to the original scale including the intercept
      // Recording only the primary phenotype
      RSS(i, j) = sum(square(Ys(0) - Xs(0) * _b_est_v1.row(0).t()));
			
			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      temp2(span(1, p)) = _b_est_v1.row(0) / Xs_s.row(0);
			xbarb = sum(_b_est_v1.row(0) / Xs_s.row(0) % Xs_c.row(0));
			MPI_Allgather(&xbarb, 1, MPI_FLOAT, allxbarb, 1, MPI_FLOAT, MPI_COMM_WORLD); //////////////////
			xbarb = 0; for (w=0; w<size; w++) xbarb += allxbarb[w];
      temp2(0) = Ys_c(0) - xbarb;
      _b_est.row(i) = temp2;
			///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      
			// Recording second phenotypes for GWAS data
			//temp2(span(1, p)) = _b_est_v1.row(1) / Xs_s.row(1);
			//temp2(0) = Ys_c(1) - sum(_b_est_v1.row(1) / Xs_s.row(1) % Xs_c.row(1));
			//_b_est_2.row(i) = temp2;
			//////////////////////////////////////////////

			_b_est_st = _b_est_v1;

		} // end for i

		/////////////////////////////////////////////////////////////
		//objval.save("./res/obj_" + cPar.penalize + "_" + patch::to_string(nGroup) + "_" + patch::to_string(ns(0)) + "_" + patch::to_string(p) + 
		// "_" + patch::to_string(j) + "_" + patch::to_string(size) + "_" + patch::to_string(rank) + ".txt", csv_ascii);
		/////////////////////////////////////////////////////////////

		_b_est.resize(i, p+1);
		b_est(j) = _b_est;

		//////////////////////////////////////////////
		_b_est_2.resize(i, p+1);
		b_est_2(j) = _b_est_2;
		//////////////////////////////////////////////
    
    lambda1(j) = lambda1(j)(span(0, i-1));
    cout << endl;
  } // end for j
  
  // Calculate AIC, BIC, GCV
  L1 = 0;
  for (i=0; i<lambda1.n_elem; i++) {
  	L1 = lambda1(i).n_elem > L1 ? lambda1(i).n_elem : L1;
  }
  nzero.set_size(L1, L2);
  RSS.resize(L1, L2);

  //nzero.fill(NA_INTEGER);
	nzero.fill(datum::nan);

  for (j=0; j<L2; j++) {
  	for (i=0; i<lambda1(j).n_elem; i++) {
  		nzero(i, j) = sum(abs(b_est(j).row(i)) > datum::eps) - 1;
  	}
  }

	// delete Variables	//////////////////////////
	delete[] alleLoop, curXsb, sumXsb, allXsb, allxbarb;
	//////////////////////////////////////////////

  // Re-estimation of the coefficients
	if (re_est == true) {
		urowvec active_set3;
	  frowvec _b_re_est_v1;
	  fmat _b_re_est;
		fmat RSS2(L1, L2);
		//RSS2.fill(NA_REAL);
		RSS2.fill(datum::nan);
		b_re_est.set_size(q);
		for (j=0; j<q; j++) {
			_b_re_est.set_size(b_est(j).n_rows, p+1);
			for (i=0; i<b_est(j).n_rows; i++) {
				active_set3 = (abs(b_est(j)(i, span(1, p))) > datum::eps);
				_Re_Estimate(Ys(0), Xs(0), _b_re_est_v1, active_set3);
				RSS2(i, j) = sum(square(Ys(0) - Xs(0) * _b_re_est_v1.t()));
				temp2(span(1, p)) = _b_re_est_v1 / Xs_s.row(0);
				temp2(0) = Ys_c(0) - sum(_b_re_est_v1 / Xs_s.row(0) % Xs_c.row(0));
				_b_re_est.row(i) = temp2;
			}
			b_re_est(j) = _b_re_est;
		}
	}

	// change to normal scale for genotype and phenotype data
	for (i=0; i<m; i++) {
		Xs(i).each_row() %= Xs_s.row(i);
		Xs(i).each_row() += Xs_c.row(i);
		Ys(i) += Ys_c(i);
	}
	/////////////////////////////////////////////////////////

	cout << "Finished!" << endl;   
	return;
} 



// Compute MSE using multiple nodes
void CTPRMpiMSE(fmat & mse, field<fmat> & b_est, field<fvec> & lambda1, 
	fvec lambda2, int L1, int L2, fvec & Ys_test, fmat & Xs_test, int size, fmat & cvR2){
	
	int i, j, k, w, nXsb=Ys_test.n_elem;
	float *curXsb = new float[nXsb];
	float *sumXsb = new float[nXsb];
	float *allXsb = new float[nXsb*size];
	fvec Xsb(nXsb);
	fvec Xsb1(nXsb);

	for (w=0; w<nXsb; w++) curXsb[w] = 0;
	for (w=0; w<nXsb; w++) sumXsb[w] = 0;

	MPI_Barrier(MPI_COMM_WORLD); // barrier synchronization
	
	for (i=0; i<L2; i++) {
		for (j=0; j<lambda1(i).n_elem; j++) {
			Xsb1 = Xs_test * b_est(i).row(j).t();
			for (w=0; w<nXsb; w++) curXsb[w] = Xsb1(w);
			MPI_Allgather(curXsb, nXsb, MPI_FLOAT, allXsb, nXsb, MPI_FLOAT, MPI_COMM_WORLD); /////////////////
			for (w=0; w<nXsb; w++) sumXsb[w] = 0;
			for (k=0; k<size; k++) for (w=0; w<nXsb; w++) sumXsb[w] += allXsb[k*nXsb+w];
			for (w=0; w<nXsb; w++) Xsb(w) = sumXsb[w];
			mse(j, i) = mean(square(Ys_test - Xsb));
			cvR2(j, i) = pow(as_scalar(cor(Ys_test, Xsb)),2);
			//cvR2(j, i) = 1.0-mse(j, i)/mean(square(Ys_test-mean(Ys_test))); ///// 
		}
	}

	delete[] curXsb, sumXsb, allXsb;
}


 
// Main procedure
void CTPRMpiTune(field<fvec> & Ys, field<fmat> & Xs, const fcube & A,
	field<fmat> & b_est, field<fmat> & b_est_2, field<fmat> & b_re_est, fmat & b_sec,  
	fvec & b_min, fvec & b_min0, fmat & cvm, fvec & ctprRes, PARAM & cPar) {

	cout << "**************************************************************" << endl;
	cout << "                COORDINATE DECENT ALGORITHM                   " << endl;
	cout << "**************************************************************" << endl;
	cout << endl; 

	int nFold = cPar.nFold; 
	int separinds = cPar.separinds; 
	int rank = cPar.rank; 
	int size = cPar.size; 

	// Estimate all parameters
	PrintCurrentTime(cPar); // Check Current Time
	CTPRMpiProc(Ys, Xs, A, cPar.lambda1, cPar.lambda2, b_est, b_est_2, b_re_est, b_sec, cPar.nzero, cPar);
	if (cPar.lambda2.n_elem==1) {
		fvec b_min_tmp; b_min_tmp = b_est(0).row(0).t();
		b_min_tmp.save(cPar.output+patch::to_string(rank)+".beta0", csv_ascii);
	}
	PrintCurrentTime(cPar); // Check Current Time

	cout << "Coordinate decent algorithm process [" << 0 << "] has been finished..." << endl;

	// Create result list
	//int q = cPar.lambda2.n_elem;
	//for (i=0; i<q; i++) b_est(i).save("./param/" + fnameXs + "b_est1_" + patch::to_string(i) + "_" + patch::to_string(size) + "_" + patch::to_string(rank) + ".txt", csv_ascii);
	//for (i=0; i<q; i++) b_est_2(i).save("./param/" + fnameXs + "b_est2_" + patch::to_string(i) + "_" + patch::to_string(size) + "_" + patch::to_string(rank) + ".txt", csv_ascii);
	//for (i=0; i<q; i++) cPar.lambda1(i).save("./param/" + fnameXs + "lambda1_" + patch::to_string(i) + "_" + patch::to_string(size) + "_" + patch::to_string(rank) + ".txt", csv_ascii);
	//cPar.lambda2.save("./param/" + fnameXs + "lambda2_" + patch::to_string(size) + "_" + patch::to_string(rank) + ".txt", csv_ascii);
	
	// Use n fold CV for selecting the tuning parameter lambda1 and lambda2 /////////////////////////////
	cout << "Conduct CV..." << endl;

	int i, j, k, cnt1, cnt2;
	int L1 = 0, L2 = cPar.lambda2.n_elem;
	field<fvec> _lambda1(L2);
	fvec _lambda2;
	fvec f_ind(Ys(0).n_elem);
	fvec f_ind_(Ys(0).n_elem);
	fvec nf = zeros<fvec>(nFold);
	field<fmat> mse(nFold);
	field<fvec> Ys_train(Ys.n_elem);
	field<fmat> Xs_train(Xs.n_elem);
	fvec Ys_test;
	fmat Xs_test;
	field<fmat> _b_est;
	field<fmat> _b_est_2, _b_re_est;
	umat _nzero;

	// Compute max length of lambda1 and lambda2
	for (i=0; i<L2; i++) if (cPar.lambda1(i).n_elem > L1) L1 = cPar.lambda1(i).n_elem;
	cout << "Maximum length of lambda1 and lambda2: " << L1 << ", " << L2 << endl;

	// Set CV index
	for (i=0; i<Ys(0).n_elem; i++) { f_ind_(i) = i % nFold; nf(f_ind_(i))++; }
	f_ind = shuffle(shuffle(f_ind_));

	//for (i=0; i<Ys(0).n_elem; i++) cout << f_ind_(i) << " "; cout << endl;
	//for (i=0; i<Ys(0).n_elem; i++) cout << f_ind(i) << " "; cout << endl;
	//for (i=0; i<nFold; i++) cout << nf(i) << endl;

	//cout << "Fold Group (Rank [" << rank << "]) "; for (i=0; i<Ys(0).n_elem; i++) cout << f_ind(i) << " "; cout << endl;
	
	// Initialize mse, Ys_train, Xs_train
	for (i=0; i<nFold; i++) {mse(i).set_size(L1, L2); mse(i).fill(datum::inf);}
	for (i=1; i<Ys.n_elem; i++) Ys_train(i) = Ys(i);
	for (i=1; i<Xs.n_elem; i++) Xs_train(i) = Xs(i);

	// Conduct CV procedure
	for (i=0; i<nFold; i++) {

		// Set lambda1 and lambda2
		_lambda2 = cPar.lambda2;
		for (j=0; j<L2; j++) _lambda1(j) = cPar.lambda1(j);

		// Set Ys_train, Ys_test
		Ys_train(0) = zeros<fvec>(Ys(0).n_elem-nf(i));
		if (!separinds) for (k=1; k<Ys.n_elem; k++) Ys_train(k) = zeros<fvec>(Ys(k).n_elem-nf(i));////
		Ys_test = zeros<fvec>(nf(i));

		cnt1 = cnt2 = 0;
		for (j=0; j<Ys(0).n_elem; j++) {
			if (i == f_ind(j)) {Ys_test(cnt1) = Ys(0)(j); cnt1++;}
			else {Ys_train(0)(cnt2) = Ys(0)(j); if (!separinds) for (k=1; k<Ys.n_elem; k++) Ys_train(k)(cnt2) = Ys(k)(j); cnt2++;}/////
		}

		// Set Xs_train, Xs_test
		Xs_train(0) = zeros<fmat>(Xs(0).n_rows-nf(i),Xs(0).n_cols);
		if (!separinds) for (k=1; k<Xs.n_elem; k++) Xs_train(k) = zeros<fmat>(Xs(k).n_rows-nf(i),Xs(k).n_cols);/////
		Xs_test = ones<fmat>(nf(i),Xs(0).n_cols+1); Xs_test /= (float)size;
		
		cnt1 = cnt2 = 0;
		for (j=0; j<Xs(0).n_rows; j++) {
			if (i == f_ind(j)) {Xs_test(cnt1,span(1,Xs(0).n_cols)) = Xs(0).row(j); cnt1++;}
			else {Xs_train(0).row(cnt2) =  Xs(0).row(j); if (!separinds) for (k=1; k<Xs.n_elem; k++) Xs_train(k).row(cnt2) =  Xs(k).row(j); cnt2++;}/////
		}

		// Conduct CV
		PrintCurrentTime(cPar); // Check Current Time
		CTPRMpiProc(Ys_train, Xs_train, A, _lambda1, _lambda2, _b_est, _b_est_2, _b_re_est, b_sec, _nzero, cPar);
		PrintCurrentTime(cPar); // Check Current Time

		cout << "Coordinate decent algorithm process [" << i+1 << "] has been finished..." << endl;

		// Compute MSE
		fmat cvR2(L1,L2); cvR2.zeros(); //////

		CTPRMpiMSE(mse(i), _b_est, _lambda1, _lambda2, L1, L2, Ys_test, Xs_test, size, cvR2);

		/*if (rank == size-1) {
			mse(i).save("./res/cvmse_" + cPar.fnameYs + "_"  + patch::to_string(cPar.maxNz) + patch::to_string(cPar.nFold) + "_"  + cPar.penalize + "_" + patch::to_string(cPar.lambda2(0)) +
				"_" + patch::to_string(i+1) + ".txt", csv_ascii); /////
			cvR2.save("./res/cvr2_" + cPar.fnameYs + "_"  + patch::to_string(cPar.maxNz) +  patch::to_string(cPar.nFold) + "_"  + cPar.penalize + "_" + patch::to_string(cPar.lambda2(0)) +
				"_" + patch::to_string(i+1) + ".txt", csv_ascii); /////
		}*/
	}

	// Evaluate the result of CV
	ComputeCVRes(b_min, b_min0, mse, b_est, cPar.lambda1, cPar.lambda2, nFold, cPar.nzero, ctprRes, cvm);

	// Save Coefficients
	cout << "Save all coefficients..." << endl;
	b_min.save(cPar.output+patch::to_string(rank)+".beta", csv_ascii);
	if (cPar.lambda2.n_elem!=1) {
		b_min0.save(cPar.output+patch::to_string(rank)+".beta0", csv_ascii);
	}
	PrintCurrentTime(cPar); // Check Current Time

}

// Compute MSE and Prediction R2 (sub)
void CTPRMpiPredProc(fvec & Ys_test, fmat & Xs_test, fvec & b_min, fvec & b_min0, 
	float & mse, float & R2, float & R2_1, float & mse0, float & R20, float & R20_1, float & slope, float & slope0, int size){

	int k, w, nXsb = Ys_test.n_elem;
	float *curXsb = new float[nXsb];
	float *sumXsb = new float[nXsb];
	float *allXsb = new float[nXsb*size];

	fmat one = ones<fmat>(Xs_test.n_rows,1); one /= (float)size;
	fmat _Xs_test = Xs_test; _Xs_test.insert_cols(0, one);
	fvec Xsb(nXsb);
	fvec Xsb0(nXsb);
	fvec _Xsb, _Xsb0;

	_Xsb = _Xs_test * b_min;
	for (w=0; w<nXsb; w++) curXsb[w] = _Xsb(w);
	MPI_Allgather(curXsb, nXsb, MPI_FLOAT, allXsb, nXsb, MPI_FLOAT, MPI_COMM_WORLD); //////////////////
	for (w=0; w<nXsb; w++) sumXsb[w] = 0;
	for (k=0; k<size; k++) for (w=0; w<nXsb; w++) sumXsb[w] += allXsb[k*nXsb+w];
	for (w=0; w<nXsb; w++) Xsb(w) = sumXsb[w];

	_Xsb0 = _Xs_test * b_min0;
	for (w=0; w<nXsb; w++) curXsb[w] = _Xsb0(w);
	MPI_Allgather(curXsb, nXsb, MPI_FLOAT, allXsb, nXsb, MPI_FLOAT, MPI_COMM_WORLD); //////////////////
	for (w=0; w<nXsb; w++) sumXsb[w] = 0;
	for (k=0; k<size; k++) for (w=0; w<nXsb; w++) sumXsb[w] += allXsb[k*nXsb+w];
	for (w=0; w<nXsb; w++) Xsb0(w) = sumXsb[w];

	mse = mean(square(Ys_test - Xsb));
	mse0 = mean(square(Ys_test - Xsb0));
	R2 = pow(as_scalar(cor(Ys_test, Xsb)),2);
	R2_1 = 1.0-mse/mean(square(Ys_test-mean(Ys_test))); 
	R20 = pow(as_scalar(cor(Ys_test, Xsb0)),2);
	R20_1 = 1.0-mse0/mean(square(Ys_test-mean(Ys_test)));
	slope = stddev(Ys_test)/stddev(Xsb)*as_scalar(cor(Ys_test, Xsb));
	slope0 = stddev(Ys_test)/stddev(Xsb0)*as_scalar(cor(Ys_test, Xsb0));

	delete[] curXsb, sumXsb, allXsb;
}

 
// Compute MSE and Prediction R2
void CTPRMpiPred(fvec & Ys_test, fmat & Xs_test, fvec & b_min, fvec & b_min0, field<fmat> & b_est, 
	fmat & cvm, fvec & ctprRes, PARAM & cPar){

	float mse, R2, R2_1, mse0, R20, R20_1, slope, slope0;
	int size = cPar.size, rank = cPar.rank;
	int i, k, w, nXsb = Ys_test.n_elem;
	float *curXsb = new float[nXsb];
	float *sumXsb = new float[nXsb];
	float *allXsb = new float[nXsb*size];

	fmat one = ones<fmat>(Xs_test.n_rows,1); one /= (float)size;
	fmat _Xs_test = Xs_test; _Xs_test.insert_cols(0, one);
	fvec Xsb(nXsb);
	fvec Xsb0(nXsb);
	fvec _Xsb, _Xsb0;

	_Xsb = _Xs_test * b_min;
	for (w=0; w<nXsb; w++) curXsb[w] = _Xsb(w);
	MPI_Allgather(curXsb, nXsb, MPI_FLOAT, allXsb, nXsb, MPI_FLOAT, MPI_COMM_WORLD); //////////////////
	for (w=0; w<nXsb; w++) sumXsb[w] = 0;
	for (k=0; k<size; k++) for (w=0; w<nXsb; w++) sumXsb[w] += allXsb[k*nXsb+w];
	for (w=0; w<nXsb; w++) Xsb(w) = sumXsb[w];

	_Xsb0 = _Xs_test * b_min0;
	for (w=0; w<nXsb; w++) curXsb[w] = _Xsb0(w);
	MPI_Allgather(curXsb, nXsb, MPI_FLOAT, allXsb, nXsb, MPI_FLOAT, MPI_COMM_WORLD); //////////////////
	for (w=0; w<nXsb; w++) sumXsb[w] = 0;
	for (k=0; k<size; k++) for (w=0; w<nXsb; w++) sumXsb[w] += allXsb[k*nXsb+w];
	for (w=0; w<nXsb; w++) Xsb0(w) = sumXsb[w];

	mse = mean(square(Ys_test - Xsb));
	mse0 = mean(square(Ys_test - Xsb0));
	R2 = pow(as_scalar(cor(Ys_test, Xsb)),2);
	R2_1 = 1.0-mse/mean(square(Ys_test-mean(Ys_test))); 
	R20 = pow(as_scalar(cor(Ys_test, Xsb0)),2);
	R20_1 = 1.0-mse0/mean(square(Ys_test-mean(Ys_test)));
	slope = stddev(Ys_test)/stddev(Xsb)*as_scalar(cor(Ys_test, Xsb));
	slope0 = stddev(Ys_test)/stddev(Xsb0)*as_scalar(cor(Ys_test, Xsb0));

	//Ys_test.save("./res/Ys_test_" + patch::to_string(mse) + ".txt", csv_ascii);
	//Xsb0.save("./res/Xsb0_" + patch::to_string(mse) + ".txt", csv_ascii);
	delete[] curXsb, sumXsb, allXsb;

	// Save Results
	ctprRes(0)=R20; ctprRes(1)=R20_1; ctprRes(2)=mse0; 
	ctprRes(6)=R2; ctprRes(7)=R2_1; ctprRes(8)=mse;

	// Compute sum of number of non-zero beta values
	int *allnzbeta = new int[size];
	int nzbe0 = (int) ctprRes(4);
	int nzbe = (int) ctprRes(11);
	MPI_Allgather(&nzbe0, 1, MPI_INT, allnzbeta, 1, MPI_INT, MPI_COMM_WORLD); //////////////////
	nzbe0 = 0; for (i=0; i<size; i++) nzbe0 += allnzbeta[i];
	MPI_Allgather(&nzbe, 1, MPI_INT, allnzbeta, 1, MPI_INT, MPI_COMM_WORLD); //////////////////
	nzbe = 0; for (i=0; i<size; i++) nzbe += allnzbeta[i];
	
	int nzbe1=0;
	fvec nzbevec(b_est(0).n_rows);
	for (i=0; i<b_est(0).n_rows; i++) {
		nzbe1 = (int) cPar.nzero(i, 0);
		MPI_Allgather(&nzbe1, 1, MPI_INT, allnzbeta, 1, MPI_INT, MPI_COMM_WORLD); //////////////////
		nzbe1 = 0; for (k=0; k<size; k++) nzbe1 += allnzbeta[k];
		nzbevec(i) = nzbe1;
	}
	delete[] allnzbeta;

	// Save results
	ctprRes(4)=nzbe0; ctprRes(11)=nzbe;
	ctprRes(13)=slope0; ctprRes(14)=slope;
	if (rank == 0) ctprRes.save(cPar.output+".res", csv_ascii);
	//if (rank == 0) ctprRes.save("./res/mpires_" + fnameYs + "_" + penalize + "_" + patch::to_string(cPar.nGroup) + "_" + patch::to_string(size) + ".txt", csv_ascii);
	
	// Print Results
	if (rank == 0) {
	  cout << "**************************************************************" << endl;
		cout << "                   PREDICTION R2 AND MSE                      " << endl;
	  cout << "**************************************************************" << endl;
		cout << endl;
		cout << "(1) Prediction Results with " << cPar.penalize << endl;
		cout << "lambda2=0" << ",lambda1=" << ctprRes(3) << ",R2=" << R20 << ",R2(by MSE)=" << R20_1 << ",MSE=" << mse0;
		cout << ",Slope=" << slope0 << ",Nzbeta=" << nzbe0 << ",cvMSE=" << ctprRes(5) << endl;
		cout << ctprRes(3) << " " << R20 << " " << R20_1 << " " << mse0 << " " << slope0 << " " << nzbe0 << " " << ctprRes(5) << endl;
		cout << endl;
		cout << "(2) Prediction Results with " << cPar.penalize << " + " << cPar.penalize2 << endl;
		cout << "lambda2=" << ctprRes(10) << ",lambda1=" << ctprRes(9) << ",R2=" << R2 << ",R2(by MSE)=" << R2_1 << ",MSE=" << mse;
		cout << ",Slope=" << slope << ",Nzbeta=" << nzbe << ",cvMSE=" << ctprRes(12) << endl;
		cout << ctprRes(9) << " " << R2 << " " << R2_1 << " " << mse << " " << slope << " " << nzbe << " " << ctprRes(12) << endl;
		cout << endl;
		cout << "(3) All Prediction Results (lambda2=" << ctprRes(10) << ")" << endl;
		cout << "lambda1 R2 OriR2 MSE Slope Nzbeta cvMSE num" << endl;
	}
	
	fvec bmin, bmin0;
	for (i=0; i<b_est(0).n_rows; i++) {
		bmin = b_est(0).row(i).t();
		bmin0 = b_est(0).row(i).t();
		CTPRMpiPredProc(Ys_test, Xs_test, bmin, bmin0, mse, R2, R2_1, mse0, R20, R20_1, slope, slope0, size); 
		if (rank == 0) {
			cout << cPar.lambda1(0)(i) << " " << R2 << " " << R2_1 << " " << mse << " " << slope << " " << nzbevec(i) << " " << cvm(i,0) << " " << cPar.flamb1+i << endl;
		}
	}
	cout << "End the CTPR Program!!!!!" << endl;
	PrintCurrentTime(cPar); // Check Current Time
}


// Main Function
int main(int argc, char **argv){
	
	// Check whether paraemters are acceptable /////////
	if (argc <= 1) {
		PrintVersion();
		return EXIT_SUCCESS;
	}
	if (argc==2 && argv[1][0] == '-' && argv[1][1] == 'l') {
		PrintLicense();
		return EXIT_SUCCESS;
	}
	if (argc==2 && argv[1][0] == '-' && argv[1][1] == 'h') {
		PrintHelp();
		return EXIT_SUCCESS;
	}
	////////////////////////////////////////////////////

	// Print CTPR Version and License //////////////////
	PrintVersion();
	PrintLicense();
	////////////////////////////////////////////////////

	// Initialize MPI Code /////////////////////////////
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
  MPI_Comm_size(MPI_COMM_WORLD, &size);
	cout << "This MPI process is " << rank << " of " << size << "..." << endl;
	////////////////////////////////////////////////////

	// Initialize parameters ///////////////////////////
	PARAM cPar;
	cPar.useMPI = 1; // use MPI
	cPar.fnameXs = ""; cPar.fextXs = ""; cPar.fnameXstest = ""; cPar.fextXstest = "";
	cPar.fnameSs = ""; cPar.fextSs = "";
	cPar.fnameYs = ""; cPar.fnameYstest = "";
	cPar.ncolXs = 0; cPar.ncolXstest = 0;
	cPar.ncolYs = 0; cPar.ncolYstest = 0;
	cPar.useSummary = 0; cPar.useTest = 0; cPar.nsecTrait = 0; cPar.useScaling = 0;
	cPar.nrowXs.set_size(2); cPar.nrowXs(0)=0;
	cPar.nrowYs.set_size(2); cPar.nrowYs(0)=0;
	cPar.nrowXstest = 0; cPar.nrowYstest = 0;
	cPar.penalize = "Lasso"; cPar.penalize2 = "CTPR";
	cPar.separinds = 0; cPar.nFold = 5;
	cPar.perc = 0.25;
	cPar.error = false;	cPar.re_est = false; cPar.slambda2 = false;
	cPar.maxNz = 0; cPar.maxIter = 10000; 
	cPar.gamma = 3.0; cPar.tol = 0.0001; 
	cPar.output = "CTPRResultMPI";
	cPar.nGroup = size; cPar.start = 1; cPar.flamb1 = 0; cPar.llamb1 = 100;
	cPar.rank = rank; cPar.size = size; cPar.ranktxt = " (Rank ["+patch::to_string(rank)+"])";
	cPar.t_start = clock(); // set current time
	////////////////////////////////////////////////////

	// Assign particular values to the paramters ///////
	if (!AssignParameters(argc, argv, cPar)) return EXIT_FAILURE;
	if (!cPar.slambda2) {
		cPar.lambda2.set_size(12);
		cPar.lambda2(0)=0; cPar.lambda2(1)=0.06109; cPar.lambda2(2)=0.13920 ; cPar.lambda2(3)=0.24257; cPar.lambda2(4)=0.38582;
		cPar.lambda2(5)=0.59756; cPar.lambda2(6)=0.94230; cPar.lambda2(7)=1.60280; cPar.lambda2(8)=3.37931; cPar.lambda2(9)=8.5;  
		cPar.lambda2(10)=15.5; cPar.lambda2(11)=24.5;
	}
	////////////////////////////////////////////////////

	// Check All datasets //////////////////////////////
	string fnameXs1, fnameXstest1, fnameSs1;
	PrintCurrentTime(cPar); // Check Current Time
	fnameXs1 = cPar.fnameXs + patch::to_string(rank+cPar.start) + "." + cPar.fextXs;
	fnameSs1 = cPar.fnameSs + patch::to_string(rank+cPar.start) + "." + cPar.fextSs;
	fnameXstest1 = cPar.fnameXstest + patch::to_string(rank+cPar.start) + "." + cPar.fextXstest;
	if (!CheckData(fnameXs1, fnameXstest1, cPar.fnameYs, cPar.fnameYstest, fnameSs1, cPar)) return EXIT_FAILURE;	
	PrintCurrentTime(cPar); // Check Current Time
	////////////////////////////////////////////////////

	// Check Parameter combination ////////////////////
	if (!CheckCombination(cPar)) return EXIT_FAILURE;
	///////////////////////////////////////////////////
	 
	// Set A matrix ////////////////////////////////////
	int i, j, k;
	fcube A; A.set_size(cPar.ncolYs+cPar.nsecTrait, cPar.ncolYs+cPar.nsecTrait, cPar.ncolXs);
	fmat A2(cPar.ncolYs+cPar.nsecTrait,cPar.ncolYs+cPar.nsecTrait); A2.fill(1); A2.diag()-=1;
	for (i=0; i<cPar.ncolXs; i++) A.slice(i) = A2;
	////////////////////////////////////////////////////

	// Read Phenotype and Genotype files and Summary files
  field<fvec> Ys(cPar.ncolYs);
	field<fmat> Xs(cPar.ncolYs);
	fvec Ystest, imr2;
	fmat Xstest, b_sec, seb_sec;
	float avgseb; 
	
	PrintCurrentTime(cPar); // Check Current Time
	LoadPheno(Ys, Ystest, cPar.fnameYs, cPar.fnameYstest, cPar); // Read Phenotype File
	LoadGeno(Xs, Xstest, fnameXs1, fnameXstest1, cPar); // Read Genotype File
	if (cPar.useSummary) {
		LoadSummary(b_sec, seb_sec, imr2, fnameSs1, cPar); // Read Summary File
		// reset A matrix
		avgseb = mean(mean(seb_sec));
		fmat A3(cPar.ncolYs+cPar.nsecTrait,cPar.ncolYs+cPar.nsecTrait); A3.fill(1); A3.diag()-=1;
		for (i=0; i<cPar.ncolXs; i++) {
			for (j=0; j<cPar.ncolYs; j++) {
				for (k=j+1; k<cPar.ncolYs; k++) {A3(j,k) = A3(k,j) = 1; /*imr2(i)/avgseb;*/}
				for (k=cPar.ncolYs; k<cPar.ncolYs+cPar.nsecTrait; k++) {A3(j,k) = A3(k,j) = avgseb/seb_sec(k-cPar.ncolYs,i); /*imr2(i)/seb_sec(k-ncolYs,i);*/}
			}
			//A.slice(i) = A3; ///
		}
	}
	PrintCurrentTime(cPar); // Check Current Time
	////////////////////////////////////////////////////

	// Scaling for secondary traits ////////////////////
	//if (!ScaleTraits(cPar)) return EXIT_FAILURE;
	////////////////////////////////////////////////////
	
	// Estimate parameters and select lambda1, lambda2 and Save beta coefficients
	field<fmat> b_est, b_est_2, b_re_est;
	fvec bmin, bmin0;
	fmat cvm;
	fvec ctprRes; ctprRes.set_size(15); // result file	
	CTPRMpiTune(Ys, Xs, A,	b_est, b_est_2, b_re_est, b_sec, bmin, bmin0, cvm, ctprRes, cPar); 
	//////////////////////////////////////////////////////

	// Compute MSE and Prediction R2
	if (cPar.useTest) CTPRMpiPred(Ystest, Xstest, bmin, bmin0, b_est, cvm, ctprRes, cPar); 
	//bmin.save("./res/b_min_" + patch::to_string(mse) + "_" + patch::to_string(rank) + ".txt", csv_ascii);
	//////////////////////////////////////////////////////

	// Finalize MPI Code  //////////////////////////////
	MPI_Barrier(MPI_COMM_WORLD); // barrier synchronization
	cout << "End ctprmpi program" << cPar.ranktxt << endl;
	MPI_Finalize();
	cout << endl;
	////////////////////////////////////////////////////

	return EXIT_SUCCESS;
}
