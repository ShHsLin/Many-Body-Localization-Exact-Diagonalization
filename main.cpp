#include<iostream>
#include<fstream>
#include"khash.h"
#include<cmath>
#include<slepceps.h>
#include<string>
#include<stdlib.h>
#include<time.h>
#include<armadillo>
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include <numeric>


#define ASC
#define LevelSpacing
#define ForceMoreState
//#define IPR_log
//#define Header
//#define SAVE_OPDM
//#define SAVE_EIG_VEC


#ifndef ASC
#include<random>
#else
#include <boost/random.hpp>
typedef boost::mt19937 base_generator_type;
#endif









////// This is a typedef for a random number generator.
////// Try boost::mt19937 or boost::ecuyer1988 instead of boost::minstd_rand


const int L_default=10;
const double t_default=1.;
const double V_default=-2;
const double W_default=1;
const double E_default=0;

const int realization_default=1;
const double rand_max=RAND_MAX;

const int num_eig_taken_default = 3; // Should be >=3


KHASH_MAP_INIT_INT64(ToLargeMap, int);
KHASH_MAP_INIT_INT64(ToSmallMap, int);


static char help[] = "This is an exact diagonalization code for many body localization in 1d. The code uses SLEPc PETSc and armadillo libraries.\n\n"
"The command line options are:\n"
"  -L <L>, where <L> = system size\n"
"  -R <R>, where <R> = number of realization\n"
"  -V <V>, where <V> = interaction strength\n"
"  -W <W>, where <W> = disorder strength\n"
"  -E <E>, where <E> = energy density\n";



long long int Factorial(int n){
	long long int sum=1;
	if (n==0) return sum;
	else if (n==1) return sum;
	else{
		while(n>0){
			sum*=n;
			n--;
		}
		return sum;
	}
}
long long int HalfFactorial(int n){
	long long int sum=1;
	for(int Top=n;Top>n/2;Top--){
		sum*=Top;
	}
	return sum;
}



int toLIndex( int chain[], int L){
	//long long int power[16]={1,4,16,64,256,1024,4096,16384,65536,262144,1048576,4194304,16777216,67108864,268435456,1073741824  };
	int LargeIndex=0;
	for(int i=0; i< L; i++){
		LargeIndex += chain[i]*pow(2,i);

	}
	return LargeIndex;
}

void toChain(const int LargeIndex, int chain[],int N){
	for(int i=0;i<N;i++){
		chain[i]=(LargeIndex/(int)pow(2,i))%2;
	};
}




bool is_file_exist(const char *fileName)
{
	std::ifstream infile(fileName);
	return infile.good();
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char** argv){


#ifndef ASC
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator (seed);
	std::uniform_real_distribution<double> distribution(0.0,1.0);
#endif

#ifdef ASC
	base_generator_type generator(static_cast<unsigned int>(std::time(0)));
	//generator.seed(static_cast<unsigned int>(std::time(0)));
	boost::uniform_real<> uni_dist(0,1);
	boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni(generator, uni_dist);
#endif

	SlepcInitialize(&argc,&argv,(char*)0,help);
	//      ierr = PetscOptionsInsertString(common_options); CHKERRQ(ierr);// this adds my defaults
	//      ierr = PetscOptionsInsert(&argc,&argv,PETSC_NULL); CHKERRQ(ierr);      // this has command line options replace defaults


	int ret;
	khiter_t k_iter;
	khash_t(ToLargeMap) *LargehashMap = kh_init(ToLargeMap);
	khash_t(ToSmallMap) *SmallhashMap = kh_init(ToSmallMap);

	PetscErrorCode ierr;
	Mat H_no_potent;
	PetscReal      error,tol,re,im;
	PetscInt     Istart,Iend,nev,maxit,its,nconv,PetscIntL=L_default, PetscIntR=realization_default;
	PetscScalar    eigMaxr,eigMaxi,eigMinr,eigMini,kr,ki;
	PetscScalar     PetscScalarW=W_default, PetscScalarV=V_default, PetscScalart=t_default, PetscScalarE=E_default;
	ST             st,stOPDM;//Spectral Transform
	KSP            ksp,kspOPDM;
	PC             pc,pcOPDM;
	PetscViewer    viewerbi, viewerasc, viewerasc2;
	PetscRandom    rctx;

	/*
	   char  common_options[]      = "-st_type sinvert -st_ksp_type preonly -st_pc_type cholesky\
	   -st_pc_factor_mat_solver_package mumps\
	   -mat_mumps_icntl_13 1";
	 */



	ierr = PetscOptionsGetReal(NULL,"-t",&PetscScalart,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-W",&PetscScalarW,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-V",&PetscScalarV,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetReal(NULL,"-E",&PetscScalarE,NULL);CHKERRQ(ierr);

	ierr = PetscOptionsGetInt(NULL,"-L",&PetscIntL,NULL);CHKERRQ(ierr);
	ierr = PetscOptionsGetInt(NULL,"-R",&PetscIntR,NULL);CHKERRQ(ierr);

	const double t=PetscScalart;
	const double W=PetscScalarW;
	const double V=PetscScalarV;
	const double E=PetscScalarE;
	const int L=PetscIntL;
	const int realization=PetscIntR;

	const int Number_ele=L/2;
	const long long int smallDim=HalfFactorial(L)/Factorial(Number_ele);
	const long long int largeDim=pow(2,L);
	const int halfChainDim=pow(2,L/2);

	int num_eig_taken;
	if (E != 0){
		num_eig_taken = num_eig_taken_default;
	}
	else{
		num_eig_taken = 3;
	}



	//int basisMatrix[smallDim][L];
	// Basis matrix map samllDim_index to system config
	int** basisMatrix= new int*[smallDim];
	for (int i=0; i< smallDim; i++) {
		basisMatrix[i]=new int[L];
	}

	double* entangleEntropy=new double[realization*num_eig_taken];
	for (int i=0; i<realization*num_eig_taken; i++) {
		entangleEntropy[i]=0;
	}

	double** occuSpecMatrix = new double*[realization*num_eig_taken];
	for (int i=0; i<realization*num_eig_taken; i++) {
		occuSpecMatrix[i]=new double[L];
		for (int j=0; j<L; j++) {
			occuSpecMatrix[i][j]=0;
		}
	}
	double* gapRatio=new double[realization*(num_eig_taken-2)];
	for (int i=0; i<realization*(num_eig_taken-2); i++) {
		gapRatio[i]=0;
	}
	double* IPR=new double[realization*num_eig_taken];
	for (int i=0; i<realization*num_eig_taken; i++) {
		IPR[i]=0;
	}


	Mat Tmatrix[L][L];





	int MPI_Rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &MPI_Rank);
	std::cout<<"\n"<<"MPIRANK:"<<MPI_Rank<<"\n";
	std::cout<<"largedim : "<<largeDim<<std::endl;
	std::cout<<"smalldim : "<<smallDim<<std::endl;


	int tempW=PetscScalarW*10;
	int tempV=PetscScalarV*10;
	int tempE=PetscScalarE*100;

	char filenumL[10]; sprintf(filenumL, "%d",L);
	char filenumV[10];
	if (tempV >= 0) {
		sprintf(filenumV, "%d",tempV);
	}
	else{
		sprintf(filenumV, "_%d",-tempV);
	}
	char filenumW[10]; sprintf(filenumW, "%d",tempW);
	char filenumE[10]; sprintf(filenumE, "%d",tempE);

	int gapRatioCount=0;
	int realizationCount=0;
	int numVecCount=1;
#ifdef SAVE_OPDM
	int numMatCount=1;
#endif
	char fileend[10] = ".data";
	char filenameH[100]="H_no_pot_L";
	char filenameVec[100]="EigVec_L";
	char filenameMat[100]="OPDM_L";
	char filenameOS[100]="OccuSpect_L";
	char filenameEE[100]="EntEntro_L";
	char filenameGR[100]="GapRatio_L";
	char filenameIPR[100]="IPR_L";

	strcat (filenameH,filenumL);
	strcat (filenameVec,filenumL);
	strcat (filenameMat,filenumL);
	strcat (filenameOS,filenumL);
	strcat (filenameEE,filenumL);
	strcat (filenameGR,filenumL);
	strcat (filenameIPR,filenumL);

	strcat (filenameH,"V");
	strcat (filenameVec,"V");
	strcat (filenameMat,"V");
	strcat (filenameOS,"V");
	strcat (filenameEE,"V");
	strcat (filenameGR,"V");
	strcat (filenameIPR,"V");

	strcat (filenameH,filenumV);
	strcat (filenameVec,filenumV);
	strcat (filenameMat,filenumV);
	strcat (filenameOS,filenumV);
	strcat (filenameEE,filenumV);
	strcat (filenameGR,filenumV);
	strcat (filenameIPR,filenumV);

	strcat (filenameVec,"W");
	strcat (filenameMat,"W");
	strcat (filenameOS,"W");
	strcat (filenameEE,"W");
	strcat (filenameGR,"W");
	strcat (filenameIPR,"W");

	strcat (filenameVec,filenumW);
	strcat (filenameMat,filenumW);
	strcat (filenameOS,filenumW);
	strcat (filenameEE,filenumW);
	strcat (filenameGR,filenumW);
	strcat (filenameIPR,filenumW);

	strcat (filenameVec,"E");
	strcat (filenameMat,"E");
	strcat (filenameOS,"E");
	strcat (filenameEE,"E");
	strcat (filenameGR,"E");
	strcat (filenameIPR,"E");

	strcat (filenameVec,filenumE);
	strcat (filenameMat,filenumE);
	strcat (filenameOS,filenumE);
	strcat (filenameEE,filenumE);
	strcat (filenameGR,filenumE);
	strcat (filenameIPR,filenumE);

	strcat (filenameH,fileend);
	strcat (filenameVec,".m");
	strcat (filenameMat,".m");
	strcat (filenameOS,".m");
	strcat (filenameEE,".m");
	strcat (filenameGR,".m");
	strcat (filenameIPR,".m");


	PetscViewerCreate(PETSC_COMM_WORLD,&viewerbi);
	PetscViewerCreate(PETSC_COMM_WORLD,&viewerasc);
	PetscViewerCreate(PETSC_COMM_WORLD,&viewerasc2);
	PetscViewerSetType(viewerbi, PETSCVIEWERBINARY);
	PetscViewerSetType(viewerasc, PETSCVIEWERASCII);
	PetscViewerSetType(viewerasc2, PETSCVIEWERASCII);


	std::cout<<"Building basisMatrix with dim : "<<smallDim<<", "<<L<<std::endl;
	{
		int testChain[L];
		int sIndex=0;	// sIndex = smallDim Index
		for(int lIndex=0 ;lIndex<largeDim ;lIndex++){
			toChain(lIndex, testChain,L);
			int sum=0;
			for(int i=0;i<L;i++){
				sum+=testChain[i];
			};
			if(sum==L/2){
				// 1. Build toLargeMap & toSmallMap 2. Build BasisMatrix
				k_iter = kh_put(ToLargeMap, LargehashMap, sIndex, &ret);
				kh_value(LargehashMap, k_iter) = lIndex;
				k_iter = kh_put(ToSmallMap, SmallhashMap, lIndex, &ret);
				kh_value(SmallhashMap, k_iter) = sIndex;
				for(int j=0;j<L;j++){
					basisMatrix[sIndex][j]=testChain[j];
				}
				sIndex++;
			}
		}
	}




	// - - - --  -- - - - - - -  - -- - - - -  -- - - - //
	std::cout<<"Building Transition Matrix"<<std::endl;
	// - - - --  -- - - - - - -  - -- - - - -  -- - - - //
	// Only the upper half of the Transition Matrix is built
	for(int i=0; i<L; i++){
		for(int j=0; j<L; j++){
			//	                ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, smallDim,smallDim,10,NULL,&Tmatrix[i][j]);CHKERRQ(ierr);

			ierr = MatCreate(PETSC_COMM_WORLD,&Tmatrix[i][j]);CHKERRQ(ierr);
			ierr = MatSetSizes(Tmatrix[i][j],PETSC_DECIDE,PETSC_DECIDE,smallDim,smallDim);CHKERRQ(ierr);
			ierr = MatSetFromOptions(Tmatrix[i][j]);CHKERRQ(ierr);
			ierr = MatSetUp(Tmatrix[i][j]);CHKERRQ(ierr);
		}
	}

	for(int j=0; j<L; j++){
		// i = i
		// Building Tmatrix[j][j]
		for(int sIndex=0;sIndex<smallDim; sIndex++){

			//tempChain = BaseStateMatrix(ind,:);
			if (basisMatrix[sIndex][j]==0){
				continue;
			}
			else{
				ierr = MatSetValue(Tmatrix[j][j],sIndex,sIndex,1,INSERT_VALUES);CHKERRQ(ierr);
			}
		}
		ierr = MatAssemblyBegin(Tmatrix[j][j],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
		ierr = MatAssemblyEnd(Tmatrix[j][j],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);


		int tempChain[L];
		for(int i=j+1; i<L; i++){
			for(int sIndex=0;sIndex<smallDim; sIndex++){
				//Copying the chain to tempchain
				for(int chainIndex=0;chainIndex<L;chainIndex++){
					tempChain[chainIndex]=basisMatrix[sIndex][chainIndex];
					//tempChain=BaseStateMatrix(ind,:);
				}
				if (tempChain[j]==0 || tempChain[i]!=0){
					continue;
				}
				else {
					int summ=1;
					for(int chainIndex=j;chainIndex<=i;chainIndex++){
						summ+=tempChain[chainIndex];
					}
					tempChain[j]=0;
					tempChain[i]=1;
					// %%tempInd= toSmallMap(toNumber(tempChain,Num_Site));

					int tempInd = toLIndex(tempChain,L);
					k_iter      = kh_get(ToSmallMap, SmallhashMap, tempInd);
					if (k_iter == kh_end(SmallhashMap)){
						std::cout<<"Error, no element in map\n";
					}
					tempInd     = kh_value(SmallhashMap, k_iter);
					if (tempInd>smallDim) { std::cout<<"Error, tempInd : "<<tempInd<<std::endl;}
					ierr = MatSetValue(Tmatrix[i][j],tempInd,sIndex,pow(-1,summ),INSERT_VALUES);
					CHKERRQ(ierr);

					// %%Tmatrix(((i-1)*Num_Site+j-1)*smallDim+tempInd,ind)=pow((-1),summ);
				}
			}
			ierr = MatAssemblyBegin(Tmatrix[i][j],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
			ierr = MatAssemblyEnd(Tmatrix[i][j],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

		}
	}


	std::cout<<" ---- The end of generating dictionary and Tmatrix ----\n";

	std::cout<<" ---- V = "<<V<<" -------- \n";



	//  We construct the reduced Hamiltonian in the subspace.
	//  We try not to construct the hamiltonian each time.
	//  First we construct the common Hamiltonian and then
	//  add the randomly generated potential at each realization.
	if( is_file_exist(filenameH) ){
		std::cout<<"Loading H_no_poten from the existed file\n";
		ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filenameH,FILE_MODE_READ,&viewerbi);
		ierr = MatCreate(PETSC_COMM_WORLD,&H_no_potent);CHKERRQ(ierr);

		MatLoad(H_no_potent,viewerbi);
		PetscViewerDestroy(&viewerbi);
	}
	else
	{
		std::cout<<"BUILDING THE H_no_poten MATRIX\n";

		ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, smallDim,smallDim,100,NULL,&H_no_potent);CHKERRQ(ierr);
		//ierr = MatCreate(PETSC_COMM_WORLD,&H_no_potent);CHKERRQ(ierr);
		//ierr = MatSetSizes(H_no_potent,PETSC_DECIDE,PETSC_DECIDE,smallDim,smallDim);CHKERRQ(ierr);
		//ierr = MatSetUp(H_no_potent);CHKERRQ(ierr);

		for(int sIndex=0; sIndex<smallDim; sIndex++){
			int tempChain[L];
			for(int i=0; i<L; i++){
				tempChain[i]=basisMatrix[sIndex][i];
			}
			int lIndex=toLIndex(tempChain,L);
			//----------------------------------------//
			// Filling in the diagonal repulsion term
			double term_repul=0;
			//for(int i=0;i<L;i++){ //PBC
			for(int i=0;i<L-1;i++){
				//term_repul  +=  V*(tempChain[i]-0.5)*(tempChain[(i+1)%L]-0.5);
				term_repul  +=  V*(tempChain[i])*(tempChain[(i+1)%L]);
			}
			ierr = MatAssemblyBegin(H_no_potent,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
			ierr = MatAssemblyEnd(H_no_potent,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
			ierr = MatSetValue(H_no_potent,sIndex,sIndex,term_repul,INSERT_VALUES);CHKERRQ(ierr);
			// --------------------------------------//

			for(int i=0;i<L;i++){  
				//for(int i=0;i<L-1;i++){ 
				// If there is particle
				if (tempChain[i]!=0){
					// Try  Hop to the i+1
					int Hop_i_mod=(i+1)%L;
					// If there is no particle at i+1
					if (tempChain[Hop_i_mod]==0){
						int new_lIndex=lIndex-pow(2,i)+pow(2,Hop_i_mod);
						k_iter      = kh_get(ToSmallMap, SmallhashMap, new_lIndex);
						if (k_iter == kh_end(SmallhashMap)){
							std::cout<<"Error, no element in map, happen at hopping i+1\n";
						}
						int new_sIndex    = kh_value(SmallhashMap, k_iter);
						//OBC
						if (Hop_i_mod==0){
							//%% A extra minus sign is taken
							//%% Due to periodic boundary condition
							//%% And it actually pass through odd number of electrons
							//%% row_H(1,toSmallMap(new_ind) )=row_H(1,toSmallMap(new_ind) )+0.5;

							////////////////////////////////////////////
							////// Comment out below while in OBC //////
							////////////////////////////////////////////

							//  ierr = MatAssemblyBegin(H_no_potent,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
							//  ierr = MatAssemblyEnd(H_no_potent,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
							//  ierr = MatSetValue(H_no_potent,sIndex,new_sIndex,1.,ADD_VALUES);CHKERRQ(ierr);
						}
						//OBC
						else{
							//ierr = MatSetValue(H_no_potent,sIndex,new_sIndex,-0.5,INSERT_VALUES);
							ierr = MatAssemblyBegin(H_no_potent,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
							ierr = MatAssemblyEnd(H_no_potent,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
							ierr = MatSetValue(H_no_potent,sIndex,new_sIndex,-1.*t,ADD_VALUES);CHKERRQ(ierr);
							// %% row_H(1,toSmallMap(new_ind) )=row_H(1,toSmallMap(new_ind) )-0.5;
						}
					}

					// %%  Hop to the i-1
					int flag=0;
					if (i!=0)
						Hop_i_mod=i-1;
					else{
						Hop_i_mod=L-1;
						flag=1;
					}
					if (tempChain[Hop_i_mod]==0){
						//new_ind=ind-2^(i-1)+2^(Hop_i_mod-1);
						int new_lIndex=lIndex-pow(2,i)+pow(2,Hop_i_mod);
						k_iter      = kh_get(ToSmallMap, SmallhashMap, new_lIndex);
						if (k_iter == kh_end(SmallhashMap)){
							std::cout<<"Error, no element in map, happen at hopping i-1\n";
						}
						int new_sIndex    = kh_value(SmallhashMap, k_iter);

						if (flag==1){
							//%% 	A extra minus sign is taken
							//%%  	Due to periodic boundary condition
							//%% 	And it actually pass through odd number of electrons
							//%%  row_H(1,toSmallMap(new_ind) )= row_H(1,toSmallMap(new_ind) )+0.5;


							////////////////////////////////////////////
							////// Comment out below while in OBC //////
							////////////////////////////////////////////

							//   ierr = MatAssemblyBegin(H_no_potent,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
							//   ierr = MatAssemblyEnd(H_no_potent,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
							//   ierr = MatSetValue(H_no_potent,sIndex,new_sIndex,1.,ADD_VALUES);CHKERRQ(ierr);
						}
						else{
							//row_H(1,toSmallMap(new_ind) )= row_H(1,toSmallMap(new_ind) )-0.5;
							//ierr = MatSetValue(H_no_potent,sIndex,new_sIndex,-0.5,INSERT_VALUES);
							ierr = MatAssemblyBegin(H_no_potent,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
							ierr = MatAssemblyEnd(H_no_potent,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
							ierr = MatSetValue(H_no_potent,sIndex,new_sIndex,-1.*t,ADD_VALUES);CHKERRQ(ierr);
						}

					}
				}
			}
			}

			ierr = MatAssemblyBegin(H_no_potent,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
			ierr = MatAssemblyEnd(H_no_potent,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
			//////////////////////////////////////////////
			//Save H_no_potent for next realization use //
			// - - - - - - - - - - - - - -  - - - - - - -
			ierr =  PetscViewerBinaryOpen(PETSC_COMM_WORLD,filenameH,FILE_MODE_WRITE,&viewerbi);CHKERRQ(ierr);
			ierr = MatView(H_no_potent,viewerbi);CHKERRQ(ierr);
			//////////////////////////////////////////////

			//////////////////////////////////////////////
			//Save H_no_potent in MatrixMarket format for GPU calculation 
			// - - - - - - - - - - - - - -  - - - - - - -			
			/*			PetscViewerASCIIOpen(PETSC_COMM_WORLD, "Htemp2" , &viewerasc);
						ierr = PetscViewerSetFormat(viewerasc, PETSC_VIEWER_ASCII_MATRIXMARKET);
						ierr = MatView(H_no_potent,viewerasc);CHKERRQ(ierr);
			 */			//////////////////////////////////////////////

			PetscViewerDestroy(&viewerbi);
		}

		std::cout<<" ---- The end of generating Hamiltonian w/o rand pot ----\n";



		//////////////////////////////////////////////////////
		//	PREPARING FOR WRITING THE EIGEN VECTOR and OPDM //
		//////////////////////////////////////////////////////

		//ierr = PetscViewerPushFormat(viewerasc, PETSC_VIEWER_ASCII_MATLAB);
		ierr = PetscViewerSetFormat(viewerasc, PETSC_VIEWER_ASCII_MATLAB);
		ierr = PetscViewerSetFormat(viewerasc2, PETSC_VIEWER_ASCII_MATLAB);
		//ierr = PetscViewerSetFormat(viewerasc, PETSC_VIEWER_ASCII_DENSE);
#ifdef SAVE_EIG_VEC
		if( is_file_exist(filenameVec) ){
			std::cout<<"File:"<<filenameVec<<"  Already Exist, Appending the data... "<<std::endl;
			PetscViewerFileSetMode(viewerasc,FILE_MODE_APPEND);
			PetscViewerFileSetName(viewerasc, filenameVec);
			//ierr =  PetscViewerASCIIOpen(PETSC_COMM_WORLD, filenameVec , &viewerasc);
		}
		else{
			std::cout<<"No File Exist, Creating "<<filenameVec<<std::endl;
			PetscViewerFileSetMode(viewerasc,FILE_MODE_WRITE);
			PetscViewerFileSetName(viewerasc, filenameVec);
			//ierr =  PetscViewerASCIIOpen(PETSC_COMM_WORLD, filenameVec , &viewerasc);
		}
#endif
#ifdef SAVE_OPDM
		if( is_file_exist(filenameMat) ){
			std::cout<<"File:"<<filenameMat<<" Already Exist, Appending the data... "<<std::endl;
			PetscViewerFileSetMode(viewerasc2,FILE_MODE_APPEND);
			PetscViewerFileSetName(viewerasc2, filenameMat);
		}
		else{
			std::cout<<"No File Exist, Creating "<<filenameMat<<std::endl;
			PetscViewerFileSetMode(viewerasc2,FILE_MODE_WRITE);
			PetscViewerFileSetName(viewerasc2, filenameMat);
		}
#endif

		/////////////////////////////////////////////////
		/////////////////////////////////////////////////

		Vec natOrbR;
		ierr=VecCreate(PETSC_COMM_WORLD,&natOrbR); CHKERRQ(ierr);
		ierr=VecSetSizes(natOrbR,PETSC_DECIDE,L); CHKERRQ(ierr);
		ierr=VecSetFromOptions(natOrbR); CHKERRQ(ierr);

		for(int realIndex=0;realIndex<realization;realIndex++){

			Mat Hamiltonian;
			Vec            xr,xi;
			EPS            eps;         /* eigenproblem solver context */
			EPSType        type;
			//PetscScalar    randnum;


			ierr= MatDuplicate(H_no_potent,MAT_COPY_VALUES,&Hamiltonian);CHKERRQ(ierr);
			//		ierr= MatConvert(H_no_potent,MATAIJ,MAT_INITIAL_MATRIX,&Hamiltonian);CHKERRQ(ierr);

			//////////////////////////////////
			//	Give randpotential	//
			//////////////////////////////////


			double randArray[L];
			for(int i=0;i<L;i++){
#ifndef ASC
				// Method 1
				randArray[i] = -W+2*W*distribution(generator);
#else
				//Method 4
				double rnum = uni();
				randArray[i] = -W+2*W*rnum;
#endif

				// Method 2
				//PetscRandomGetValue(rctx,&randnum);CHKERRABORT(PETSC_COMM_WORLD,ierr);
				//double rnum=(double)randnum;

				//std::cout<<"Rand:"<< rnum<<std::endl;
				//std::cout<<"Rand:"<< randArray[i]<<std::endl;
			}

			//for(int site_i=0; site_i<L; site_i++){
			//	randArray[site_i] = pow(-1,site_i)*0.0001;
			//}			 


			ierr = MatAssemblyBegin(Hamiltonian,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
			ierr = MatAssemblyEnd(Hamiltonian,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);

			for(int sIndex =0;sIndex<smallDim;sIndex++){
				double potential_term=0;
				for(int i=0;i<L;i++){
					//potential_term += randArray[i]*(basisMatrix[sIndex][i]-0.5);
					potential_term += randArray[i]*(basisMatrix[sIndex][i]);
					//dia = (chain-0.5)*Rand_W;
				}
				// %%    reduce_Ham(i,i )=reduce_Ham(i,i )+dia;
				ierr = MatSetValue(Hamiltonian,sIndex,sIndex,potential_term,ADD_VALUES);
				CHKERRQ(ierr);
			}

			ierr = MatAssemblyBegin(Hamiltonian,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
			ierr = MatAssemblyEnd(Hamiltonian,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

			//////////////////////////////////
			//	Solve the Ax=kx		//
			//	Matrix diagonalization	//
			//////////////////////////////////

			ierr = MatCreateVecs(Hamiltonian,NULL,&xr);CHKERRQ(ierr);
			ierr = MatCreateVecs(Hamiltonian,NULL,&xi);CHKERRQ(ierr);

			/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
			   Create the eigensolver and set various options
			   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

			//  Create eigensolver context
			ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);
			//   Set operators. In this case, it is a standard eigenvalue problem
			ierr = EPSSetOperators(eps,Hamiltonian,NULL);CHKERRQ(ierr);
			ierr = EPSSetProblemType(eps,EPS_HEP);CHKERRQ(ierr);


			/* - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
			   Solving the smallest EigenValue
			   - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
			ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL);CHKERRQ(ierr);

			/*			ierr = EPSSetWhichEigenpairs(eps,EPS_ALL);CHKERRQ(ierr);
						EPSSetInterval(eps,-20,20);
						ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
						ierr = STSetType(st,STSINVERT);CHKERRQ(ierr);
						ierr = STGetKSP(st,&ksp);CHKERRQ(ierr);
						ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
						ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
						ierr = PCSetType(pc,PCCHOLESKY);CHKERRQ(ierr);*/
			//ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

#ifdef ForceMoreState
			//ierr = EPSSetDimensions(eps,3,PETSC_DEFAULT,PETSC_DEFAULT);
			ierr = EPSSetDimensions(eps,num_eig_taken,PETSC_DEFAULT,PETSC_DEFAULT);	
#endif
			ierr = EPSSolve(eps);CHKERRQ(ierr);

			if (E !=0) {



				ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);
				ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %D\n\n",nconv);CHKERRQ(ierr);

				if (nconv>0) {
					ierr = PetscPrintf(PETSC_COMM_WORLD,
							"           k          ||Ax-kx||/||kx||\n"
							"   ----------------- ------------------\n");CHKERRQ(ierr);
					int i=0;

					ierr = EPSGetEigenpair(eps,i,&eigMinr,&eigMini,xr,xi);CHKERRQ(ierr);
					ierr = EPSComputeError(eps,i,EPS_ERROR_RELATIVE,&error);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
					re = PetscRealPart(eigMinr);
					im = PetscImaginaryPart(eigMinr);
#else
					re = eigMinr;
					im = eigMini;
#endif
					if (im!=0.0) {
						ierr = PetscPrintf(PETSC_COMM_WORLD," %9f%+9f j %12g\n",(double)re,(double)im,(double)error);
						CHKERRQ(ierr);
					} else {
						ierr = PetscPrintf(PETSC_COMM_WORLD,"   %12f     %12g\n",(double)re,(double)error);
						CHKERRQ(ierr);
					}

					ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
				}


				/* - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
				   Solving the largest EigenValue
				   - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

				ierr = EPSSetWhichEigenpairs(eps,EPS_LARGEST_REAL);CHKERRQ(ierr);
				//   Set solver parameters at runtime
				//		ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);
				ierr = EPSSolve(eps);CHKERRQ(ierr);


				/*
				   ierr = EPSGetIterationNumber(eps,&its);CHKERRQ(ierr);
				   ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);CHKERRQ(ierr);
				   ierr = EPSGetType(eps,&type);CHKERRQ(ierr);
				   ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
				   ierr = EPSGetDimensions(eps,&nev,NULL,NULL);CHKERRQ(ierr);
				   ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRQ(ierr);
				   ierr = EPSGetTolerances(eps,&tol,&maxit);CHKERRQ(ierr);
				   ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit);CHKERRQ(ierr);
				 */
				ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);
				ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %D\n\n",nconv);CHKERRQ(ierr);


				if (nconv>0) {
					ierr = PetscPrintf(PETSC_COMM_WORLD,
							"           k          ||Ax-kx||/||kx||\n"
							"   ----------------- ------------------\n");CHKERRQ(ierr);
					int i=0;
					ierr = EPSGetEigenpair(eps,i,&eigMaxr,&eigMaxi,xr,xi);CHKERRQ(ierr);
					ierr = EPSComputeError(eps,i,EPS_ERROR_RELATIVE,&error);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
					re = PetscRealPart(eigMaxr);
					im = PetscImaginaryPart(eigMaxr);
#else
					re = eigMaxr;
					im = eigMaxi;
#endif
					if (im!=0.0) {
						ierr = PetscPrintf(PETSC_COMM_WORLD," %9f%+9f j %12g\n",(double)re,(double)im,(double)error);
						CHKERRQ(ierr);
					} else {
						ierr = PetscPrintf(PETSC_COMM_WORLD,"   %12f     %12g\n",(double)re,(double)error);
						CHKERRQ(ierr);
					}

					ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
				}



				/* - - - - -- - - - - - - - - - - - - - - - - - - - - - - - -
				   Solving the EigenVector
				   - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - */

				//-eps_interval -0.1,0.0 -st_type sinvert -st_ksp_type preonly -st_pc_type cholesky   -st_pc_factor_mat_solver_package mumps -mat_mumps_icntl_13 1

				PetscScalar eigTarget;
				eigTarget = eigMinr+(eigMaxr-eigMinr)*E/2;
				std::cout<<" target : "<<eigTarget<<"\n";
				ierr = EPSSetTarget(eps,eigTarget );CHKERRQ(ierr);
				//ierr = EPSSetInterval(eps,(eigMinr+eigMaxr)/2-0.1,(eigMinr+eigMaxr)/2+0.1);CHKERRQ(ierr);
				ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
				//ierr = STSetShift(ST st,PetscScalar shift);CHKERRQ(ierr);

				ierr = STSetType(st,STSINVERT);CHKERRQ(ierr);
				ierr = STGetKSP(st,&ksp);CHKERRQ(ierr);
				ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
				ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
				ierr = PCSetType(pc,PCCHOLESKY);CHKERRQ(ierr);
				ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);CHKERRQ(ierr);
				/*		PetscInt ival;
						PetscBool isset;
						ierr = PetscOptionsGetInt(PETSC_NULL, "-mat_mumps_icntl_13",&ival,&isset);CHKERRQ(ierr);
						if (isset==PETSC_FALSE) {
						ierr = PetscOptionsSetValue("-mat_mumps_icntl_13","1");
						std::cout<<"SET -mat_mumps_icntl_13 to 1\n";
						CHKERRQ(ierr);
						}
				 */
				//std::cout<<"Search near : "<<(eigMinr+eigMaxr)/2<<" +- 0.1 \n";

				ierr = EPSSetWhichEigenpairs(eps,EPS_TARGET_REAL);CHKERRQ(ierr);
#ifdef ForceMoreState
				//ierr = EPSSetDimensions(eps,3,PETSC_DEFAULT,PETSC_DEFAULT);
				ierr = EPSSetDimensions(eps,num_eig_taken,PETSC_DEFAULT,PETSC_DEFAULT);	
#endif
				//
				ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);
				KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);
				ierr = EPSSolve(eps);CHKERRQ(ierr);


			}
			else{
				//////////////////////////////////////////////////////////////////////
				// Solving for the ground state case, i.e. smallest Eigenvalue only.
				//////////////////////////////////////////////////////////////////////
			}

			ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);
			ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %D\n\n",nconv);CHKERRQ(ierr);
			if (nconv>0) {
				double arr_eigenvalues[num_eig_taken];
				ierr = PetscPrintf(PETSC_COMM_WORLD,
						"           k          ||Ax-kx||/||kx||\n"
						"   ----------------- ------------------\n");CHKERRQ(ierr);
				//for (int i=0;i<nconv;i++){
				for (int i=0;i<num_eig_taken;i++){

					//   Get converged eigenpairs: i-th eigenvalue is stored in kr (real part) and
					//   ki (imaginary part)
					ierr = EPSGetEigenpair(eps,i,&kr,&ki,xr,xi);CHKERRQ(ierr);
					//   Compute the relative error associated to each eigenpair
					ierr = EPSComputeError(eps,i,EPS_ERROR_RELATIVE,&error);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
					re = PetscRealPart(kr);
					im = PetscImaginaryPart(kr);
#else
					re = kr;
					im = ki;
#endif
					if (im!=0.0) {
						ierr = PetscPrintf(PETSC_COMM_WORLD," %9f%+9f j %12g\n",(double)re,(double)im,(double)error);
						CHKERRQ(ierr);
					} else {
						ierr = PetscPrintf(PETSC_COMM_WORLD,"   %12f     %12g\n",(double)re,(double)error);
						CHKERRQ(ierr);
					}

					ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);


					arr_eigenvalues[i]=re;	

					PetscScalar vecNorm;
					ierr = VecNorm(xi,NORM_INFINITY,&vecNorm);CHKERRQ(ierr);
					if(vecNorm==0){
						char VecCount[10];
						sprintf(VecCount,"%d",numVecCount);
						char numVec[100] = "Vec(";
						strcat (numVec,VecCount);
						strcat (numVec,",:)");
						numVecCount++;
#ifdef SAVE_EIG_VEC
						ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, filenameVec , &viewerasc);
						//ierr = PetscViewerPushFormat(viewerasc, PETSC_VIEWER_ASCII_MATLAB);
						//ierr = PetscViewerSetFormat(viewerasc, PETSC_VIEWER_ASCII_MATLAB);	
						ierr = PetscObjectSetName((PetscObject)xr, numVec);
						ierr = VecView(xr,viewerasc);CHKERRQ(ierr);
						PetscViewerDestroy(&viewerasc);
#endif
					}
					else {
						std::cout<<"eigenvector has imaginary part !!! \n";
					}

					//  SKIP !!! SKIP !!!! SKIP !!!!! //	
					if (i>0) continue;
					//  SKIP !!! SKIP !!!! SKIP !!!!! //	



					///////////////////////////////////////////////////
					//      Calculating Half -Chain Entropy		//
					///////////////////////////////////////////////////
					/*
					   - Calculate the basis for the full system
					   L=4: |1100>, |1010>, |1001>, |0110>, |0101>, |0011>

					   - Calculate the subbasis states for both parts of the system (since its half of the system you only need one set)
					   L=4: |11>, |10>, |01>, |00>

					   - Run over the full basis and build a map which maps the index to the two indizes in the subbasis sets
					   i_{full} -> map[i] = (i_1,i_2)
					   L=4: map = [(0,3),(1,1),(1,2),(2,1),(2,2),(3,0)]

					   - The reduced density matrix is calculated as
					   \rho_{i_1,i_1'} = \sum_{i_2} \psi_{i_1,i_2}^* \psi_{i_1',i_2} |i_1><i_1'|
					   |\psi> = \sum_{i_{full}} \psi_{i_{full}} |i_{full}>
					   = \sum_{i_1,i_2} \psi_{i_1,i_2} |i_1>|i_2>

					   -> The \psi_{i_1,i_2} can be obtained from the map directly as this gives you the two substates
					 */

					//1.)
					//2.) map is constructed at beginning.
					//3.) two for loop --> index, one for loop --> sum
					{
						/*
						//double reduced_rho[halfChainDim][halfChainDim];
						double** reduced_rho=new double*[halfChainDim];
						for(int j=0;j<halfChainDim;j++){
						reduced_rho[j]=new double[halfChainDim];
						}

						for(int j=0;j<halfChainDim; j++){
						for( int k=j;k<halfChainDim; k++){
						reduced_rho[j][k]=0;
						// \rho_{i_1,i_1'} = \sum_{i_2} \psi_{i_1,i_2}^* \psi_{i_1',i_2} |i_1><i_1'|
						for (int l=0; l<halfChainDim; l++) {
						// j --> i_1, k --> i_1', l --> i_2
						int index1= j*halfChainDim + l ;//\psi_{i_1,i_2}
						int index2= k*halfChainDim + l ;//\psi_{i_1',i_2}

						k_iter      = kh_get(ToSmallMap, SmallhashMap, index1);
						if (k_iter == kh_end(SmallhashMap)){
						continue;
						//Finding index1 Error, no element in map\n";
						}
						int smallindex1     = kh_value(SmallhashMap, k_iter);

						k_iter      = kh_get(ToSmallMap, SmallhashMap, index2);
						if (k_iter == kh_end(SmallhashMap)){
						continue;
						//Finding index1 Error, no element in map\n";
						}
						int smallindex2     = kh_value(SmallhashMap, k_iter);


						PetscScalar temp1;
						PetscScalar temp2;

						ierr = VecGetValues(xr,1,&smallindex1,&temp1); CHKERRQ(ierr);
						ierr = VecGetValues(xr,1,&smallindex2,&temp2); CHKERRQ(ierr);
						//if( temp1>0.01){
						//	printf("j=%d,k=%d,l=%d, temp1: %16f, temp2: %16f \n",j,k,l,temp1,temp2);
						//}
						reduced_rho[j][k]+= temp1*temp2;
						}
						//End of the summation over i_2
						}
						}

						using namespace arma;
						mat rho(halfChainDim, halfChainDim);
						for (int j=0; j<halfChainDim; j++) {
						for (int k=0; k<j ;k++){
						rho(j,k)=reduced_rho[k][j];
						}
						for (int k=j; k<halfChainDim; k++) {
						rho(j,k)=reduced_rho[j][k];
						}

						}
						std::cout<<"trace of rho: "<<trace(rho)<<std::endl;


						for(int j=0;j<halfChainDim;j++){
						delete[] reduced_rho[j];
						}
						delete[] reduced_rho;
						 */
						////////////////////////////////////////////////////////////////////////////////////
						using namespace arma;
						mat PSI(halfChainDim, halfChainDim);
						for (int j=0; j<halfChainDim; j++) {
							for (int k=0; k<halfChainDim ;k++){
								int index1= j*halfChainDim + k ;//\psi_{j,k}
							k_iter      = kh_get(ToSmallMap, SmallhashMap, index1);
							if (k_iter == kh_end(SmallhashMap)){
								PSI(j,k)=0;
								continue;
								// index not found, no element in map\n";
							}
							int smallindex1     = kh_value(SmallhashMap, k_iter);
							PetscScalar temp1;
							ierr = VecGetValues(xr,1,&smallindex1,&temp1); CHKERRQ(ierr);	
							PSI(j,k)=temp1;
							//printf("vector element: %f\n",temp1);
							}
						}
						mat rho = PSI.t()*PSI;
						std::cout<<"trace of rho: "<<trace(rho)<<std::endl;

						/////////////////////////////////////////////////////////////////////////////////////////////////


						vec eigval;
						mat eigvec;
						bool flag = eig_sym(eigval, eigvec, rho,"dc");
						if (flag==true){
							double Entropy=0;//=arma::trace(rhologrho );
							for (int j=0; j<halfChainDim; j++) {
								if (eigval(j)>pow(10,-8)) {
									//std::cout<<"Singular Value: "<<eigval(j)<<"\n";
									Entropy	-= eigval(j)*log(eigval(j));
								}
							}
							std::cout<<"Entropy : "<<Entropy<<"\n";
							entangleEntropy[realizationCount]=Entropy;
						}
						else continue;
					}
















					//////////////////////////////////
					//      Generating OPDM		//
					//////////////////////////////////

					Vec     yr,yi;
					Mat	OPDM;
					PetscInt  nconv2;
					PetscScalar OPDMValue;
					PetscScalar temp1;

					ierr = MatCreateVecs(Tmatrix[0][0],NULL,&yr);CHKERRQ(ierr);
					ierr = MatCreateVecs(Tmatrix[0][0],NULL,&yi);CHKERRQ(ierr);

					ierr = MatCreate(PETSC_COMM_WORLD,&OPDM);CHKERRQ(ierr);
					ierr = MatSetSizes(OPDM,PETSC_DECIDE,PETSC_DECIDE,L,L);CHKERRQ(ierr);
					//ierr = MatSetType(OPDM,MATDENSE );
					//ierr = MatSetOption(OPDM, MAT_HERMITIAN, PETSC_TRUE);CHKERRQ(ierr);
					//ierr = MatSetFromOptions(OPDM);CHKERRQ(ierr);
					ierr = MatSetUp(OPDM);CHKERRQ(ierr);

					ierr = VecNorm(xi,NORM_INFINITY,&vecNorm);CHKERRQ(ierr);
					if(vecNorm==0){
						for(int j=0;j<L;j++){
							//std::cout<<"Crash inside i="<<i<<"j="<<j<<" loop\n";
							//ierr = VecDot(yr,xi,&OPDMValue);CHKERRQ(ierr); //x'*y
							//PetscComplex number = OPDMValue*PETSC_i;
							//ierr = MatSetValue(OPDM,j,j,number,ADD_VALUES);CHKERRQ(ierr);

							for(int k=j+1;k<L;k++){
								// OPDM(k,j)= X' * Tmatrix[k][j] * X;	
								ierr = MatMult(Tmatrix[k][j],xr,yr);CHKERRQ(ierr); // y=T*x
								ierr = VecDot(yr,xr,&OPDMValue);CHKERRQ(ierr); //x'*y
								ierr = MatSetValue(OPDM,k,j,OPDMValue,INSERT_VALUES);CHKERRQ(ierr);
							}
						}


						ierr = MatAssemblyBegin(OPDM,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
						ierr = MatAssemblyEnd(OPDM,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

						Mat OPDMtrans;
						ierr = MatTranspose(OPDM, MAT_INITIAL_MATRIX,&OPDMtrans);CHKERRQ(ierr);
						//MatHermitianTranspose
						ierr = MatAYPX(OPDM,(PetscScalar)1,OPDMtrans,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
						ierr = MatDestroy(&OPDMtrans);CHKERRQ(ierr);

						ierr = MatAssemblyBegin(OPDM,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
						ierr = MatAssemblyEnd(OPDM,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
						MatSetOption(OPDM, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

						for(int j=0;j<L;j++){

							ierr = MatMult(Tmatrix[j][j],xr,yr);CHKERRQ(ierr); // y=T*x
							ierr = VecDot(yr,xr,&OPDMValue);CHKERRQ(ierr); //x'*y
							ierr = MatSetValue(OPDM,j,j,OPDMValue,ADD_VALUES);CHKERRQ(ierr);

						}
					}
					else {
						std::cout<<"eigenvector has imarginary part !!! \n";
					}

					//            OPDM=OPDM+triu(OPDM',1);


					ierr = MatAssemblyBegin(OPDM,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
					ierr = MatAssemblyEnd(OPDM,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

					/////////////////////////////////////
					//      SAVING  OPDM               //
					/////////////////////////////////////
#ifdef SAVE_OPDM
					Mat denseOPDM;
					ierr = MatConvert(OPDM, MATDENSE ,MAT_INITIAL_MATRIX, &denseOPDM);CHKERRQ(ierr);
					char MatCount[10];
					sprintf(MatCount,"%d",numMatCount);
					char numMat[100] = "Mat(";
					strcat (numMat,MatCount);
					strcat (numMat,",:,:)");
					ierr = PetscObjectSetName((PetscObject)denseOPDM, numMat);
					ierr = MatView(denseOPDM,viewerasc2);CHKERRQ(ierr);
					ierr = MatDestroy(&denseOPDM);CHKERRQ(ierr);
					numMatCount++;
#endif

					//////////////////////////////////
					//	Diag OPDM		//
					//////////////////////////////////
					EPS epsOPDM;
					//  Create eigensolver context
					ierr = EPSCreate(PETSC_COMM_WORLD,&epsOPDM);CHKERRQ(ierr);
					//   Set operators. In this case, it is a standard eigenvalue problem
					ierr = EPSSetOperators(epsOPDM,OPDM,NULL);CHKERRQ(ierr);
					ierr = EPSSetProblemType(epsOPDM,EPS_HEP);CHKERRQ(ierr);
					ierr = EPSSetInterval(epsOPDM,-0.1,1.1);CHKERRQ(ierr);
					ierr = EPSSetWhichEigenpairs(epsOPDM,EPS_ALL);CHKERRQ(ierr);

					ierr = EPSGetST(epsOPDM,&stOPDM);CHKERRQ(ierr);
					//ierr = STSetShift(ST st,PetscScalar shift);CHKERRQ(ierr);

					ierr = STSetType(stOPDM,STSINVERT);CHKERRQ(ierr);
					ierr = STGetKSP(stOPDM,&kspOPDM);CHKERRQ(ierr);
					ierr = KSPGetPC(kspOPDM,&pcOPDM);CHKERRQ(ierr);
					ierr = KSPSetType(kspOPDM,KSPGMRES);CHKERRQ(ierr);
					ierr = PCSetType(pcOPDM,PCCHOLESKY);CHKERRQ(ierr);
					//ierr = PCSetType(pcOPDM,PCBJACOBI);CHKERRQ(ierr);
					ierr = PCFactorSetMatSolverPackage(pcOPDM,MATSOLVERMUMPS);CHKERRQ(ierr);


					ierr = EPSSolve(epsOPDM);CHKERRQ(ierr);

					ierr = EPSGetConverged(epsOPDM,&nconv2);CHKERRQ(ierr);
					ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %D\n\n",nconv2);CHKERRQ(ierr);
					if (nconv2>0) {
						for (int j=0;j<nconv2;j++){
							ierr = EPSGetEigenpair(epsOPDM,j,&kr,&ki,natOrbR,NULL);CHKERRQ(ierr);
							re = kr;
							im = ki;
							ierr = PetscPrintf(PETSC_COMM_WORLD,"   %12f     %12g\n",(double)re,(double)error);
							occuSpecMatrix[realizationCount][j]=(double)re;                         
#ifdef IPR_log 
							for (int k=0;k<L;k++){
								ierr = VecGetValues(natOrbR,1,&k,&temp1); CHKERRQ(ierr);
								IPR[realizationCount] += pow((double) temp1,4)/(double)L;
							}
#endif
						}
					}
					realizationCount++;

					ierr = MatDestroy(&OPDM);CHKERRQ(ierr);
					ierr = VecDestroy(&yr);CHKERRQ(ierr);
					ierr = VecDestroy(&yi);CHKERRQ(ierr);

					ierr = EPSDestroy(&epsOPDM);CHKERRQ(ierr);

				}//End for loop num_eig_taken

#ifdef LevelSpacing
				std::vector<double> vect_eig (arr_eigenvalues, arr_eigenvalues+num_eig_taken);
				std::sort (vect_eig.begin(), vect_eig.end());
				double mean_gap_ratio=0.;
				for(int i=0; i<num_eig_taken-2; i++){
					//////
					//   Calculating Consecutive level spacing ratio, 
					//   r= min(\delta^n, \delta^{n+1}) / max(\delta^n, \delta^{n+1})
					//////
					double del_n  = vect_eig[i+1]-vect_eig[i];
					double del_n1 = vect_eig[i+2]-vect_eig[i+1];
					double gapratio=del_n/del_n1;
					//std::cout<<" E0: "<<vect_eig[i+2]<<" E1: "<<vect_eig[i+1]<<" E2:"<<vect_eig[i]<<"\n";
					//std::cout<<" deln: "<<del_n<<" deln1: "<<del_n1<<"\n";
					if(gapratio > 1){
						gapratio= del_n1/del_n;
					}
					mean_gap_ratio+=gapratio;
					//std::cout<<" Gap Ratio: "<<gapratio <<std::endl;
					//gapRatio[gapRatioCount] = gapratio;
					//gapRatioCount++;
					//////////////////////////////////////////////////
					//  End of calculating level spacing ratio
					//////////////////////////////////////////////
				}//End second for loop num_eig_taken
				gapRatio[gapRatioCount] = mean_gap_ratio/(num_eig_taken-2.);
				gapRatioCount++;
				std::cout<<" MEAN GAP RATIO : "<<mean_gap_ratio/(num_eig_taken-2);
#endif
			}//End if nconv>0


			ierr = EPSDestroy(&eps);CHKERRQ(ierr);
			ierr = VecDestroy(&xr);CHKERRQ(ierr);
			ierr = VecDestroy(&xi);CHKERRQ(ierr);
			ierr = MatDestroy(&Hamiltonian);CHKERRQ(ierr);



			}


			//End of the realization//
			PetscViewerDestroy(&viewerasc);
			PetscViewerDestroy(&viewerasc2);


			//PetscRandomDestroy(&rctx);

			////////////////////////
			// Garbage Collection //
			////////////////////////

			///////////////////////////////////////////
			// - - --  Destroying the matrix  - - -- //
			///////////////////////////////////////////

			for(int i=0; i<L; i++){
				for(int j=0; j<L; j++){
					ierr = MatDestroy(&Tmatrix[i][j]);CHKERRQ(ierr);
				};
			}
			ierr = MatDestroy(&H_no_potent);CHKERRQ(ierr);

			ierr = SlepcFinalize();
			// - - --  END of SLEPc ------          //
			kh_destroy(ToSmallMap, SmallhashMap);
			kh_destroy(ToLargeMap, LargehashMap);


			FILE* fp = fopen(filenameOS, "a");
#ifdef Header
			fprintf(fp,"occuSpecMatrix=[\n");
#endif
			for (int i=0; i<realization*num_eig_taken; i++) {
				if (occuSpecMatrix[i][L-1]==0) {
					break;
				}
				else{
					for (int j=0; j<L; j++) {
						//std::cout<<occuSpecMatrix[i][j]<<" ";
						fprintf(fp,"%.*e ",16,occuSpecMatrix[i][j]);
					}
					fprintf(fp,";\n");
				}
			}
#ifdef Header
			fprintf(fp,"];");
#endif	
			fclose(fp);


			FILE* fp2 = fopen(filenameEE, "a");
#ifdef Header
			fprintf(fp2,"entangleEntropy=[\n");
#endif

			for (int i=0; i<realization*num_eig_taken; i++) {
				if (entangleEntropy[i]==0) {
					break;
				}
				else{
					fprintf(fp2,"%.*e ",16,entangleEntropy[i]);
					fprintf(fp2,"\n");
				}
			}
#ifdef Header
			fprintf(fp2,"];");
#endif
			fclose(fp2);

#ifdef LevelSpacing
			FILE* fp3 = fopen(filenameGR, "a");
#ifdef Header
			fprintf(fp3,"gapRatio=[\n");
#endif                  
			for (int i=0; i<realization*(num_eig_taken-2); i++) {
				if (gapRatio[i]==0) {
					break;
				}
				else{
					fprintf(fp3,"%.*e ",16,gapRatio[i]);
					fprintf(fp3,"\n");
				}
			}
#ifdef Header
			fprintf(fp3,"];");
#endif                  
			fclose(fp3);
#endif


#ifdef IPR_log
			FILE* fp4 = fopen(filenameIPR, "a");
#ifdef Header
			fprintf(fp4,"IPR=[\n");
#endif
			for (int i=0; i<realization*num_eig_taken; i++) {
				if (IPR[i]==0) {
					break;
				}
				else{
					fprintf(fp4,"%.*e ",16,IPR[i]);
					fprintf(fp4,"\n");
				}
			}
#ifdef Header
			fprintf(fp4,"];");
#endif
			fclose(fp4);
#endif


/*
                        FILE* fp5 = fopen("basisMatrix", "a");
                        for (int i=0; i<smallDim; i++) {
				for (int l=0; l<L; l++){
                                        fprintf(fp5,"%d",basisMatrix[i][l]);
				}
				fprintf(fp5,"\n");
                        }
                        fclose(fp5);
*/


			for (int i=0; i< smallDim; i++) {
				delete [] basisMatrix[i];
			}
			delete [] basisMatrix;
			delete [] IPR;
			delete [] entangleEntropy;
			delete [] gapRatio;

			for (int i=0; i<realization*num_eig_taken; i++) {
				delete [] occuSpecMatrix[i];
			}
			delete [] occuSpecMatrix;


			return 0;
		}


