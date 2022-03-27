//   file : hmm.cc
// version: 1.03 [August 21, 1995]
/* Copyright (C) 1994 Richard Myers and James Whitson
   Permission is granted to any individual or institution to use, copy, or
   redistribute this software so long as all of the original files are 
   included unmodified, that it is not sold for profit, and that this 
   copyright notice is retained.
*/

/*  *************************************************************
    Updated by Emdad Khan (Computer & Information Science dept,
    U.C Santa Cruz and National Semiconductor, Santa Clara, CA).
    Date 5/12/95. Brief description of the changes follows:

    (1)
    The original model written by Richard & James seemed to works for equal
    length string only. For unequal length strings (i.e vq coded words),
    i.e when different people say the same words, the program does not 
    converge (basically oscillates). The authors used "instanteneous"
    updates for each strings for the unequal case. Such scheme nullifies
    the learning a string by another etc. The commonly used approach
    is the "batch" update: a good reference is Rabiner's book or paper
    (as alos referenced by the authors). 
     
    So, I updated the code (training routine and other associated routines)
    to use batch update for unequal length strings. More detailed descrip-
    tion of the changes are given in corresponding routines i.e
    batch_train, beta_I, compute_gamma, alpha_F and reestimate routines.

    (2) 
    Some of equations related to training (namely, gamma, beta, alpha)
    showed discrepencies compared to the equations given in HMM literatures.
    Those equations are corrected and details of the changes are shown in
    corresponding routines. These routines are basically same as the 
    routines mentioned above.

    (3) 
    Some of the parameters were initialised both in the main routines as 
    well as in the function declarations in the .h file. Some of the 
    compilers does not like this. So, initialization of parameters are
    changed so that it is present only in one place. 
    
    *******************************************************************
*/ 
 
/*  *************************************************************
    Updated by Arthur Stephens (Arthur.Stephens@src.bae.co.uk)
    Date 8/30/95.

    Purified code, plugged memory leaks, corrected some out of bounds
    array conditions and got it to compile on Solaris. (Thanks Art --rm)
*/


#include <iostream.h>
#include <fstream.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "hmm.h"
#include "random.h"

/************************* HMM DECLARATION ****************************/

// create hmm, initialize model parameters from file.
//
HMM::HMM(char* filename)
{
  // set max_states, max_symbols and model parameters from file
  load_model(filename);
  
  cout << "\nUsing predefined model...\n";
  // show_probs();
}

// create hmm, initialize model parameters with random values.
//
HMM::HMM(int symbols, int states, int new_seed)
{
  if (new_seed != -1)
    set_seed(new_seed);

  // set globals
  max_symbols = symbols;
  max_states = states;

  // allocate space for model parameters
  alloc_model_matrices();

  // rnd initialize the trans and emit probs
  rnd_init_hmm_probs();

  cout << "\nUsing random model...\n";
  // show_probs();
}

// delete hmm, delete arrays and matrixes.
//
HMM::~HMM()
{
  // delete state and some numerator matrixes
  for (int i=0; i < max_states; i++) {
    delete [] b_numer_sum_recur[i];
    delete [] b_numer_sum_next[i];
    delete States[i];
  }
  delete [] b_numer_sum_recur;
  delete [] b_numer_sum_next;
  delete [] States;

  // delete strings matrixes
  for (i=0; i<num_strings; i++) {
    delete [] strings[i];
  }
  delete [] strings;
  delete [] string_len;

  // delete alpha,beta and gamma matrixes
  for (i=0; i < trellis_width; i++) {
    delete [] alpha[i];
    delete [] beta[i];
    delete [] gamma_next[i];
    delete [] gamma_recur[i];
  }
  delete [] alpha;
  delete [] beta;
  delete [] gamma_next;
  delete [] gamma_recur;

  // delete numerator, denominator and scaling_factors arrays
  delete [] a_numer_sum_recur;
  delete [] a_numer_sum_next;
  delete [] a_denom_sum_recur;
  delete [] a_denom_sum_next;
  delete [] scaling_factors;
}

// train hmm from sequence file until the sum of the probability
// changes is less than min_delta_psum.
// *** Changes by Emdad Khan: the code for the unequal length strings
// is changed to batch mode. It does not work in nonbatch mode.
// The equations given in Rabiner and elsewhere also indicates batch 
// mode training. Yi[] and yi[][] are used to add the batch mode
// update. ***

void HMM::batch_train(char *filename, double min_delta_psum) 
{
  int symbol_count,str;

  // local arrays and integers used in the case with strings of unequal lengths
  double *y1, *y2, *y3, *y4; 
  double **y5, **y6;
  int i,j;

  // dynamic allocation of arrays (ANSI)
  y1 = new double [max_states];
  y2 = new double [max_states];
  y3 = new double [max_states];
  y4 = new double [max_states];
  y5 = new double* [max_states];
  y6 = new double* [max_states];

  for (i=0 ; i < max_states; i++) {
    y5[i] = new double [max_symbols];
    y6[i] = new double [max_symbols];
  }


  // load string_matrix and set globals
  symbol_count = load_string_matrix(filename);

  // define trellis dimensions
  trellis_width = symbol_count+1;

  // allocate space for all the training arrays
  alloc_training_matrices();

  // loop until local min in probs
  double prob_sum=999999.0; // magic number,ick
  double alpha_tot,last_alpha_tot;

  for (str=0,alpha_tot=0.0; str<num_strings; str++)
    alpha_tot+=test(strings[str],string_len[str]);
  cout << "\nlog(alpha)\tprob_sum\n";
  cout << "----------\t--------\n";
  cout << alpha_tot <<"\n";

  // the following code does batch update for each case:
  // diff_len == true and != true 
  while (prob_sum > min_delta_psum) {
    if (diff_len==TRUE)  {
      last_alpha_tot = alpha_tot;
      for (i=0 ; i < max_states; i++) {
        y1[i] =0;
        y2[i] =0;
        y3[i] =0;
        y4[i] =0;

	for (int k=0; k < max_symbols; k++) {
	  y5[i][k] =0;
	  y6[i][k] =0;
	}
      }

      for (str=0, alpha_tot=0.0; str<num_strings; str++) {
	set_ab_counts(strings[str],string_len[str]);
	for (i=0,j=1; j < max_states; i++,j++) {

	  y1[i] += a_numer_sum_recur[i];
          y2[i] += a_numer_sum_next[i]; 
	  y3[i] += a_denom_sum_recur[i]; 
	  y4[i] += a_denom_sum_next[i]; 

	  for (int k=0; k < max_symbols; k++) {
	    y5[i][k] += b_numer_sum_recur[i][k]; 
	    y6[i][k] += b_numer_sum_next[i][k];
	  }
	}

	// compute recurrent probs for last state
	i = max_states-1;
	y1[i] += a_numer_sum_recur[i];      
	y3[i] += a_denom_sum_recur[i];
	for (int k=0; k < max_symbols; k++)
	  y5[i][k] += b_numer_sum_recur[i][k];
      }
      // end of top for loop

      // Transferring data to proper arrays/variables to be used 
      // in the reestimate routine
      for (i=0,j=1; j < max_states; i++,j++) {
	a_numer_sum_recur[i] = y1[i];
	a_numer_sum_next[i] = y2[i]; 
	a_denom_sum_recur[i] = y3[i]; 
	a_denom_sum_next[i]= y4[i]; 

	for (int k=0; k < max_symbols; k++) {
	  b_numer_sum_recur[i][k] = y5[i][k]; 
	  b_numer_sum_next[i][k] = y6[i][k];
	}
      }

      // compute recurrent probs for last state
      i = max_states-1;
      a_numer_sum_recur[i] = y1[i];
      a_denom_sum_recur[i] = y3[i];
      for (int k=0; k < max_symbols; k++)
	b_numer_sum_recur[i][k] = y5[i][k];

      // restimate routine initializes the prob_sum
      prob_sum = reestimate();

      for (str=0, alpha_tot=0.0; str<num_strings; str++) 
	alpha_tot+=test(strings[str],string_len[str]);     
    }
    // equal length case
    else {
      last_alpha_tot = alpha_tot;
      set_cumulative_ab_counts();
      prob_sum = reestimate();
      for (str=0,alpha_tot=0.0; str<num_strings; str++)
	alpha_tot+=test(strings[str]);
    }
    cout << alpha_tot << "\t";
    cout << prob_sum << "\n";
    if (alpha_tot < last_alpha_tot) {
      cerr << "\tWARNING: alpha values reversed!\n"; // should not occur!
      // break;
    }
  }

  cout << "\nDone training model " << filename << "...\n";
  // show_probs();

  // freeing the space used by y5 & y6 .. 
  for (i=0 ; i < max_states; i++) {
    delete [] y5[i]; // removed &'s 8/17/95
    delete [] y6[i];
  }
  delete [] y1;
  delete [] y2;
  delete [] y3;
  delete [] y4;
  delete [] y5;
  delete [] y6;
}  

// test hmm from sequence file.
//
double HMM::batch_test(char *filename)
{
  int symbol_count;
  FILE* ofp; // output file to write alpha values

  ofp = fopen("alphaout", "w");
  // load string_matrix and set globals
  symbol_count = load_string_matrix(filename);

  // define trellis dimensions
  trellis_width = symbol_count+1;

  // create space for testing strs
  alloc_testing_matrices();

  // loop through all test strings
  cout << "\ntest on strings...\n";
  double total=0.0,val;
  for (int i=0; i<num_strings; i++) {
    if (diff_len==TRUE) 
      val = test(strings[i],string_len[i]);
    else
      val = test(strings[i]);
    cout << "alpha for string[" << i << "] = " << val << " ";
    if (diff_len==TRUE)
      cout << " len= " << string_len[i];
    cout << "\n";
    fprintf(ofp, " %f", val); // writing alpha values to file
    if (i == (num_strings - 1))
    fprintf(ofp, "\n");
    total+=val;
  }
  fclose(ofp);
  cout << "------------ \nbatch alpha = "<<total<<"\n";

  cout << "\nDone testing model on" << filename << "...\n"; 
return(total);
}

// output the model parameters.
//
void HMM::show_probs()
{
  cout.precision(6);
  cout.setf(ios::fixed,ios::floatfield);

  cout << "\n\nstates: " << max_states;
  cout << "\nsymbols: " << max_symbols << "\n";
  for (int s=0; s < max_states; s++) {
    cout << States[s]->set_recur_trans() << " ";
    for (int k=0; k < max_symbols; k++)
      cout << States[s]->set_recur_out(k) << " ";
    cout << "\n";
    cout << States[s]->set_next_trans() << " ";
    for (k=0; k < max_symbols; k++)
      cout << States[s]->set_next_out(k) << " ";
    cout << "\n";
  }
  cout << "\n";
}

// output the model parameters to a file.
//
void HMM::dump_model(char* filename)
{
  ofstream model(filename);
  model.precision(6);
  model.setf(ios::fixed,ios::floatfield);

  if (!model) {
    cerr << "ERROR: Couldn't create file for model dump.";
    exit(-1);
  }

  model << "states: " << max_states << "\n";
  model << "symbols: " << max_symbols << "\n";
  for (int s=0; s < max_states; s++) {
    model << States[s]->set_recur_trans() << "\t";
    for (int k=0; k < max_symbols; k++)
      model << States[s]->set_recur_out(k) << "\t";
    model << "\n";
    model << States[s]->set_next_trans() << "\t";
    for (k=0; k < max_symbols; k++)
      model << States[s]->set_next_out(k) << "\t";
    model << "\n\n";
  }
  model << "\n";
  model.close();
  cout << "\nDumped model to file ==> " << filename << "\n";
}

// generate a random sequence according to model parameters.
//
void HMM::dump_seq(ofstream &seq, int length)
{
  int j, cur_state,k,symbol;
  double prob_trans,prob_out,accum;

  cur_state=0;
  for (j=0; j < length; j++) {
    // fill in random
    prob_trans = double(random() % 100000) / 100000.0;    
    prob_out = double(random() % 100000) / 100000.0;    
    if (prob_trans < States[cur_state]->set_recur_trans()) {
      accum = 0.0;
      symbol=max_symbols-1;
      for (k=0; k < max_symbols; k++) {
	if (prob_out < States[cur_state]->set_recur_out(k) + accum) {
	  symbol = k;
	  break;
	}
	else
	  accum += States[cur_state]->set_recur_out(k);
      }
      seq << symbol << " ";
    }
    else {
      accum = 0;
      symbol=max_symbols-1;
      for (int k=0; k < max_symbols; k++) {
	if (prob_out < States[cur_state]->set_next_out(k) + accum) {
	  symbol = k;
	  break;
	}
	else
	  accum += States[cur_state]->set_next_out(k);
      }
      seq << symbol << " ";
      cur_state++;
    }
  }

  if (cur_state < max_states-1) 
    cerr << "WARNING: Sequence did not reach final state!\n";
  seq << "\n";
}

/************************* SPACE ALLOCATION ****************************/

// allocate space for state probs.
//
void HMM::alloc_model_matrices() {

  // check params
  if (max_states <= 0 || max_symbols <= 0) {
    cerr <<"ERROR: alloc_model_matrices(), must define global sizes\n";
    exit(-1);
  }

  States = new STATE* [max_states];

  for (int i=0; i < max_states; i++) {
    States[i] = new STATE(max_symbols);
  }
}

// allocate space for training trellises.
//
void HMM::alloc_training_matrices()
{
  // check parms
  if (max_states <= 0 || max_symbols <= 0 || trellis_width <=0 ) {
    cerr <<"ERROR: alloc_training_matrices(), must define global sizes\n";
    exit(-1);
  }
  
  // space for calculation tables
  // two dimensional array decls
  alpha = new double* [trellis_width];
  beta = new double* [trellis_width];
  gamma_next = new double* [trellis_width];
  gamma_recur = new double* [trellis_width];
 
 for (int i=0; i < trellis_width; i++) {
    alpha[i] = new double [max_states];
    beta[i] = new double [max_states];
    gamma_next[i] = new double [max_states];
    gamma_recur[i] = new double [max_states];
  }

  // scaling factors 
  // fixed [trellis_width-1] (rem) 8/17/95
  scaling_factors = new double[trellis_width];

  // re-estimation sums
  a_numer_sum_next = new double[max_states];
  a_numer_sum_recur = new double[max_states];
  a_denom_sum_next = new double[max_states];
  a_denom_sum_recur = new double[max_states];
  b_numer_sum_next = new double* [max_states];
  b_numer_sum_recur = new double* [max_states];
  for (i=0; i < max_states; i++) {
    b_numer_sum_next[i] = new double [max_symbols];
    b_numer_sum_recur[i] = new double [max_symbols];
  }
}

// allocate space for testing trellises.
//
void HMM::alloc_testing_matrices()
{
  // check params
  if (max_states <= 0 || max_symbols <= 0 || trellis_width <= 0 ) {
    cerr <<"ERROR: alloc_testing_matrices(), must define global sizes\n";
    exit(-1);
  }

  // space for calculation tables
  // two dimensional array decls
  alpha = new double* [trellis_width];
 
  for (int i=0; i < trellis_width; i++) {
    alpha[i] = new double [max_states];
  }

  // scaling factors
  scaling_factors = new double[trellis_width];
}

/************************* MODEL INITIALIZATION ****************************/

// randomly initialize the model parameters
//
void HMM::rnd_init_hmm_probs()
{
  double prob_trans, *prob_syms;

  // init rnd gen
  srandom(seed);

  // dynamic allocation of arrays (ANSI)
  prob_syms = new double [max_symbols];

  for (int i=0; i < max_states; i++)
    {
      prob_trans = double(random() % 100) / 100.0;    
      rnd_sym_probs(prob_syms);
      States[i]->set_recur_trans(prob_trans);
      for (int j=0; j < max_symbols; j++)
	States[i]->set_recur_out(j,prob_syms[j]);
      rnd_sym_probs(prob_syms);
      States[i]->set_next_trans(1.0-prob_trans);
      for (j=0; j < max_symbols; j++)
	States[i]->set_next_out(j,prob_syms[j]);
    }
  States[max_states-1]->set_next_trans(0.0); // no next transition

  delete [] prob_syms;
}

// set output symbol probabilities.
//
void HMM::rnd_sym_probs(double prob_syms[])
{
  // determine the trans and out probs
  double prob_sum = 0.0;
  for (int s=0; s < max_symbols; s++) {
    prob_syms[s] = double(random() % 100) / 100.0;
    prob_sum += prob_syms[s];
  }
  for (s=0; s < max_symbols; s++)  // normalize so add to one
    prob_syms[s] = prob_syms[s] / prob_sum;
}

/************************* FILE OPERATIONS ****************************/

// set model parameters from a file.
//
void HMM::load_model(char* filename)
{
  //read in param file
  FILE *param_file;
  double prob;

  param_file = fopen(filename,"r");
  if (!param_file) {
    cerr << "ERROR: can not open " << filename << ".\n";
    exit(-1);
  }

  if (fscanf(param_file,"states: %d\n",&max_states)!=1) {
    cerr << "ERROR: Problem reading states: field of " << filename << "\n";
    exit(-1);
  }
    
  if (fscanf(param_file,"symbols: %d\n",&max_symbols)!=1 ) {
    cerr << "ERROR: Problem reading symbols: field of " << filename << "\n";
    exit(-1);
  }

  // allocate space for model parameters
  alloc_model_matrices();

  for (int i=0; i < max_states; i++) {
    fscanf(param_file,"%lf ",&prob);
    States[i]->set_recur_trans(prob);
    for (int j=0; j < max_symbols; j++) {
      fscanf(param_file,"%lf ",&prob);
      States[i]->set_recur_out(j,prob);
    }
    fscanf(param_file,"\n");

    fscanf(param_file,"%lf ",&prob);
    States[i]->set_next_trans(prob);
    for (j=0; j < max_symbols; j++) {
      fscanf(param_file,"%lf ",&prob);
      States[i]->set_next_out(j,prob);
    }
    fscanf(param_file,"\n");
  }
  fclose(param_file);
}

// figure out number of symbols and number of strings in file
// and load strings matrix and string_len array.
//
int HMM::load_string_matrix(char* filename)
{
  int str,sym,tmp,val;
  int symbol_count;

  // open file
  FILE *from;
  char line[MAX_LINE];
  from = fopen(filename,"r");

  symbol_count=0;
  for (num_strings=0; fscanf(from,"%[^\n]\n",line)!=EOF; num_strings++) {
    for (sym=0,tmp=99; tmp>1; sym++) {
      tmp=sscanf(line,"%d %[0-9 ]",&val,line);
    }
    if (symbol_count < sym)
      symbol_count=sym;
  }
  cout << "\nmax sequence length = " << symbol_count << "\n";
  cout << "number of strings = " << num_strings << "\n";
  rewind(from);
  
  // allocate matrix
  strings = new int* [num_strings];
  string_len = new int [num_strings];
  for (int i=0; i<num_strings; i++) {
    strings[i]=new int [symbol_count];
    for (sym=0; sym<symbol_count; sym++) // initialize array to -1
      strings[i][sym]=-1;
  }

  // load values into matrix
  diff_len=FALSE;
  int tmp_max_symbols=0;
  for (str=0; fscanf(from,"%[^\n]\n",line)!=EOF; str++) {
    for (sym=0,tmp=99; tmp>1; sym++) {
      tmp=sscanf(line,"%d %[0-9 ]",&val,line);
      strings[str][sym]=val;
      if (tmp_max_symbols<val+1)
	tmp_max_symbols=val+1;
    }
    string_len[str]=sym;
    if (sym!=symbol_count)
      diff_len=TRUE;
  }

  if (diff_len == TRUE)
    cout << "\nStrings of different lengths\n";

  // change max_symbols based on length of file
  if (tmp_max_symbols > max_symbols) {
    cout << "\nsetting max_symbols from file to "<<tmp_max_symbols<<"\n";
    max_symbols=tmp_max_symbols;
  }
  fclose(from);

  return symbol_count;
}

/************************* CALCULATIONS ****************************/

// rescale alpha values (from Rabiner).
// 
void HMM::rescale_alphas(int col)
{
  scaling_factors[col] = 0.0;
  for (int i=0; i < max_states; i++) {
    scaling_factors[col] += alpha[col][i];
  }
  // rescale all the alpha's and smooth them
  for (i=0; i <  max_states; i++) {
    alpha[col][i] /= scaling_factors[col];
  }
}

// rescale beta values after rescaling alphas.
//
void HMM::rescale_betas(int col)
{
  // rescale all the beta's w alpha's factors
  for (int i=0; i < max_states; i++) {
    beta[col][i] /= scaling_factors[col];
  }
}

// calculate alpha trellis and return final alpha value (recognition prob).
// 
double HMM::alpha_F(int* symbol_array, int symbol_count)
{
  double accum;
  int i,j;

  symbol_count=(symbol_count<0?trellis_width-1:symbol_count);

  // clear the trellis and set the first column
  for (i=0; i < trellis_width; i++)
    for (j=0; j < max_states; j++) {
      alpha[i][j] = 0.0;
    }
  alpha[0][0] = 1.0;  // first state starts with alpha = 1.0
  rescale_alphas(0);  // first col rescale is div by one

  // calculate the alphas for rest of cols
  for (i = 0; i < symbol_count; i++) { 		
    for (j=max_states-1; j > 0; j--) {
      accum = alpha[i][j] * 
	States[j]->set_recur_trans() *
	  States[j]->set_recur_out(symbol_array[i]);
      alpha[i+1][j] = alpha[i][j-1] * 
	States[j-1]->set_next_trans() * 
	  States[j-1]->set_next_out(symbol_array[i]) + 
	    accum;
    }
    alpha[i+1][0] = alpha[i][0] * States[0]->set_recur_trans() *
      States[0]->set_recur_out(symbol_array[i]);
    rescale_alphas(i+1);
  }

  return (alpha[symbol_count][max_states-1]);
}

// calculate beta trellis and return initial beta value,
// instead of going bottom to top, go top to bottom; the bottom
// is the exception this time.  we look at transistions from current
// state (single) to the next states (plural).
//
double HMM::beta_I(int* symbol_array, int symbol_count)
   {
     double accum;

     symbol_count=(symbol_count<0?trellis_width-1:symbol_count);     

     // clear the trellis and set the last column
     for (int i=0; i < trellis_width; i++)
       for (int j=0; j < max_states; j++) {
	 beta[i][j] = 0.0;
       }
     beta[symbol_count][max_states-1] = 1.0;  /* final beta is 1.0 */
     rescale_betas(symbol_count); // this is really div by 1 so no effect

     // begin setting all the cols except last one
     for (int t = symbol_count-1; t >= 0; t--) {
       for (int j=0; j < max_states-1; j++) {
	 accum = beta[t+1][j] * 
	   States[j]->set_recur_trans() *
	     States[j]->set_recur_out(symbol_array[t]);
	 beta[t][j] = beta[t+1][j+1] * 
	   States[j]->set_next_trans() * 
	     States[j]->set_next_out(symbol_array[t]) + 
	       accum;
       }

       beta[t][max_states-1] = beta[t+1][max_states-1] *
	 States[max_states-1]->set_recur_trans() *
	   States[max_states-1]->set_recur_out(symbol_array[t]);
       rescale_betas(t);
     }

     return (beta[0][0]);
   }

// compute gamma values and fill gamma tables for entire trellis.
//
void HMM::compute_gamma(int* symbol_array, int symbol_count)
{
  symbol_count=(symbol_count<0?trellis_width-1:symbol_count);

  // clear gammas
  for (int i=0; i < symbol_count; i++)
    for (int j=0; j < max_states; j++) {
	gamma_recur[i][j] = 0.0;
	gamma_next[i][j] = 0.0;
      }
  // calc the gamma table
  for (int t = 0; t < symbol_count; t++) {
    for (int i=0,j=1; i < max_states-1; i++,j++) {
      gamma_next[t][i] = (alpha[t][i]*
			  States[i]->set_next_trans() *
			  States[i]->set_next_out(symbol_array[t]) *
			  beta[t+1][j]) /
			    alpha[symbol_count][max_states-1];
      gamma_recur[t][i] = (alpha[t][i]* 
			   States[i]->set_recur_trans() * 
			   States[i]->set_recur_out(symbol_array[t]) *
			   beta[t+1][i]) /
			     alpha[symbol_count][max_states-1];  
    }
    gamma_recur[t][max_states-1] = (alpha[t][max_states-1] * 
				    States[max_states-1]->set_recur_trans() * 
		       States[max_states-1]->set_recur_out(symbol_array[t]) *
			 	    beta[t+1][max_states-1]) /
				      alpha[symbol_count][max_states-1];
    gamma_next[t][max_states-1] = 0.0;
  }
}


// this is the numerator of the a_ij reestimate and denom of b_ij reestimate,
// this func assumes gammas have been calc'd.
//
double HMM::a_numer(int i, int j, int symbol_count)
{
  double sum = 0.0;

  symbol_count=(symbol_count<0?trellis_width-1:symbol_count);

  for(int t=0; t < symbol_count; t++)
    {
      if (i==j) {
	sum += gamma_recur[t][i];
      }
      else if ( i < max_states-1 ) {
	sum += gamma_next[t][i];
      } else {
	cerr << "WARNING: gamma_next[t]["<<i<<"] shouldn't be requested\n";
	break;
      }
    }
  return sum;
}

// this is the denominator of the a_ij reestimate.
//
double HMM::a_denom(int i, int j, int symbol_count)
{
  symbol_count=(symbol_count<0?trellis_width-1:symbol_count);

  // j not used in this func - pass for consistency
  double sum = 0.0;
  for(int t=0; t < symbol_count; t++)
    if (i<max_states-1)
      sum += gamma_recur[t][i] + gamma_next[t][i];
    else
      sum += gamma_recur[t][i];

  return sum;
}

// this is the numerator of the b_ij reestimate.
// 
double HMM::b_numer(int i, int j, int sym, int *symbol_array,
		    int symbol_count)
{
  symbol_count=(symbol_count<0?trellis_width-1:symbol_count);

  double sum = 0.0;
  for(int t=0; t < symbol_count; t++) {
    if ( symbol_array[t] == sym ) 
      {
	if (i==j)
	  sum += gamma_recur[t][i];
	else  {
	  if (i < max_states-1)
	    sum += gamma_next[t][i];
	  else {
	    cerr << "WARNING: gamma_next[t]["<<i<<"] shouldn't be requested\n";
	    break;
	  }
	}
      }
  }
  return sum;
}

// set cumulative ab counts from an array of equal length strings.
//
double HMM::set_cumulative_ab_counts()
{
  int *symbol_array;
  double alpha_tot;

  // dynamic allocation of arrays (ANSI)
  symbol_array = new int [trellis_width-1];

  // clear cumulative sum arrays
  for(int i=0; i < max_states; i++) {
    a_numer_sum_recur[i] = 0.0;
    a_denom_sum_recur[i] = 0.0;
    a_numer_sum_next[i] = 0.0;
    a_denom_sum_next[i] = 0.0;
    for (int j=0; j < max_symbols; j++) {
      b_numer_sum_recur[i][j] = 0.0;
      b_numer_sum_next[i][j] = 0.0;
    }
  }

  alpha_tot=0.0;
  // loop all the strings calc a,b sums
  for(int s=0; s < num_strings; s++)
    {
      // set the symbol array to string from matrix
      for (int i=0; i < trellis_width-1; i++) {
	symbol_array[i] = strings[s][i];
      }

      // fill the alpha matrix
      alpha_tot += alpha_F(symbol_array);

      // fill the beta matrix
      beta_I(symbol_array);

      // fill the gamma_next and gamma_recur matrices
      compute_gamma(symbol_array);

      int j;
      for (i=0,j=1; j < max_states; i++,j++) {
	  a_numer_sum_recur[i] += a_numer(i,i);
	  a_numer_sum_next[i] += a_numer(i,j);
	  a_denom_sum_recur[i] += a_denom(i,i);
	  a_denom_sum_next[i] += a_denom(i,j);
	  for (int k=0; k < max_symbols; k++) {
	    b_numer_sum_recur[i][k] += b_numer(i,i,k,symbol_array);
	    b_numer_sum_next[i][k] += b_numer(i,j,k,symbol_array);
	  }
	}

      // compute recurrent probs for last state
      i = max_states-1;
      a_numer_sum_recur[i] += a_numer(i,i);
      a_denom_sum_recur[i] += a_denom(i,i);
      for (int k=0; k < max_symbols; k++)
	b_numer_sum_recur[i][k] += b_numer(i,i,k,symbol_array);
    }

  delete [] symbol_array;

  return(alpha_tot);
}

// set ab count from one string.
//
double HMM::set_ab_counts(int* symbol_array, int symbol_count)
{
  double alpha_tot;
  int i,j;

  // fill the alpha matrix
  alpha_tot = alpha_F(symbol_array,symbol_count);
  
  // fill the beta matrix
  beta_I(symbol_array,symbol_count);

  // fill the gamma_next and gamma_recur matrices
  compute_gamma(symbol_array,symbol_count);

  for (i=0,j=1; j < max_states; i++,j++) {
    a_numer_sum_recur[i] = a_numer(i,i,symbol_count);
    a_numer_sum_next[i] = a_numer(i,j,symbol_count);
    a_denom_sum_recur[i] = a_denom(i,i,symbol_count);
    a_denom_sum_next[i] = a_denom(i,j,symbol_count);
    for (int k=0; k < max_symbols; k++) {
      b_numer_sum_recur[i][k] = b_numer(i,i,k,symbol_array,symbol_count);
      b_numer_sum_next[i][k] = b_numer(i,j,k,symbol_array,symbol_count);
    }
  }

  // compute recurrent probs for last state
  i = max_states-1;
  a_numer_sum_recur[i] = a_numer(i,i,symbol_count);
  a_denom_sum_recur[i] = a_denom(i,i,symbol_count);
  for (int k=0; k < max_symbols; k++)
    b_numer_sum_recur[i][k] = b_numer(i,i,k,symbol_array,symbol_count);

  // delete [] symbol_array;

  return(alpha_tot);
}

// reestimate parameters after calculating ab_counts
//
double HMM::reestimate()
{
  double prob, diff_sum = 0.0;

  // loop all the transition probs

  for (int i=0, j=1; j < max_states; i++, j++)
    {
      // do all the prob transitions except last one
      prob = a_numer_sum_recur[i] / a_denom_sum_recur[i];
      diff_sum += fabs(prob - States[i]->set_recur_trans());
      States[i]->set_recur_trans(prob);

      for (int k=0; k < max_symbols; k++) {
	prob = b_numer_sum_recur[i][k] / a_numer_sum_recur[i];
	diff_sum += fabs(prob -	States[i]->set_recur_out(k));
	States[i]->set_recur_out(k,prob);
      }
      // do all the next transitions
      prob = a_numer_sum_next[i] / a_denom_sum_next[i];
      diff_sum += fabs(prob - States[i]->set_next_trans());
      States[i]->set_next_trans(prob);

      for (k=0; k < max_symbols; k++) {
	prob = b_numer_sum_next[i][k] / a_numer_sum_next[i];
	diff_sum += fabs(prob - States[i]->set_next_out(k));
	States[i]->set_next_out(k,prob);
      }
    }
  // calc the recurrent prob for last state
  i = max_states-1;
  prob = a_numer_sum_recur[i] / a_denom_sum_recur[i];
  diff_sum += fabs(prob - States[i]->set_recur_trans());
  States[i]->set_recur_trans(prob);

  for (int k=0; k < max_symbols; k++) {
    prob = b_numer_sum_recur[i][k] / a_numer_sum_recur[i];
    diff_sum += fabs(prob - States[i]->set_recur_out(k));
    States[i]->set_recur_out(k,prob);
  }

  return diff_sum;
}

// test model for a single string.
//
double HMM::test(int* string, int symbol_count)
{
  double log_alpha;

  symbol_count=(symbol_count<0?trellis_width-1:symbol_count);

  // fill alpha trellis and scaling coeffs
  log_alpha=log(alpha_F(string,symbol_count));

  // calc log probability from coeffs
  if (scaling_factors[0]>0.0) { // if scaling_factors set
    log_alpha=0.0;
    for (int t=0; t < symbol_count+1; t++)
      log_alpha += log(scaling_factors[t]);
  }
  return log_alpha;
}

/************************* STATE DECLARATIONS ****************************/

// create state object.
//
STATE::STATE(int num_symbols)
{
  recur_out = new double [num_symbols];
  next_out = new double [num_symbols];
}

// set/return the probability of generating a particluar symbol 
// during a recurrent transition. (smooth with MIN_PROB)
//
double STATE::set_recur_out(int symbol, double prob)
{
  if (prob != -1.0) {
    recur_out[symbol] = prob;
  }
  if (recur_out[symbol] < MIN_PROB)
    recur_out[symbol] = MIN_PROB;
  return recur_out[symbol];
}

// set/return the probability of generating a particluar symbol 
// during a transition to the next state. (smooth with MIN_PROB)
//
double STATE::set_next_out(int symbol, double prob)
{
  if (prob != -1.0) {
    next_out[symbol] = prob;
  }
  if (next_out[symbol] < MIN_PROB)
    next_out[symbol] = MIN_PROB;
  return next_out[symbol];
}

// set/return the probability of making a recurrent transition.
// (smooth with MIN_PROB)
//
double STATE::set_recur_trans(double prob)
{
  if (prob != -1.0) {
    recur_trans = prob;
  }
  if (recur_trans < MIN_PROB)
    recur_trans = MIN_PROB;
  return recur_trans;
}

// set/return the probability of making transitioning to the next state.
// (smooth with MIN_PROB)
//
double STATE::set_next_trans(double prob)
{
  if (prob != -1.0) {
    next_trans = prob;
  }
  if (next_trans < MIN_PROB)
    next_trans = MIN_PROB;
  return next_trans;
}

