//   file : hmm.h
// version: 1.03 [August 21, 1995]
/*
  Copyright (C) 1994 Richard Myers and James Whitson
  Permission is granted to any individual or institution to use, copy, or
  redistribute this software so long as all of the original files are 
  included unmodified, that it is not sold for profit, and that this 
  copyright notice is retained.
*/

// some global constants
const double MIN_PROB = 0.000001;	// never let alpha or beta equal zero
const int MAX_LINE = 1000;		// length of sequences
const int TRUE = 1;               	// flag
const int FALSE = 0;              	// flag

class STATE {
  double* recur_out; // array of output symbols
  double* next_out;  // array of output symbols
  double recur_trans;
  double next_trans;
  int max_symbols;
public:
  STATE(int num_symbols);
 ~STATE() {delete recur_out; delete next_out;};
  double set_recur_out(int symbol, double prob=-1.0);
  double set_next_out(int symbol, double prob=-1.0);
  double set_recur_trans(double prob=-1.0);
  double set_next_trans(double prob=-1.0);
}; 

class HMM {
  int seed;			    // seed for rnd init of probs
  int max_states;                   // number of states in the model
  int max_symbols;                  // symbol emissions per state
  int trellis_width;                // cols allocated in trellis =num_symbols-1
  int num_strings;                  // number of training/testing strings
  int** strings;                    // matrix of training/testing strings
  int diff_len;			    // TRUE if strings are of different lengths
  int* string_len;		    // use if strings are of different length
  STATE** States;                   // matrix of transition and emission probs
  double** alpha;                   // matrix for alpha trellis
  double** beta;                    // matrix for beta trellis
  double*  scaling_factors;         // array of alpha and beta scaling factors
  double** gamma_next;              // matrix of gamma transition probs
  double** gamma_recur;             // matrix of gamma recurrence probs
  double* a_numer_sum_recur;        // array of numerators for a_ij's
  double* a_numer_sum_next;         // array of numerators for a_ij's
  double* a_denom_sum_recur;        // array of denomonators for a_ij's
  double* a_denom_sum_next;         // array of denomonators for a_ij's
  double** b_numer_sum_recur;       // array of numerators for b_ij's
  double** b_numer_sum_next;        // array of numerators for b_ij's

  //	----- SPACE ALLOC -----
  void alloc_model_matrices();
  void alloc_training_matrices();
  void alloc_testing_matrices();

  //	------- MODEL INIT -------
  void rnd_init_hmm_probs();
  void rnd_sym_probs(double prob_syms[]);

  //	------- FILE OPS -------
  void load_model(char* filename);
  int load_string_matrix(char* filename);

  //	------ CALCULATIONS -------
  void rescale_alphas(int col);
  void rescale_betas(int col);
  double alpha_F(int* symbol_array, int symbol_count=-1);
  double beta_I(int* symbol_array, int symbol_count=-1);
  void compute_gamma(int* symbol_array, int symbol_count=-1);
  double a_numer(int i, int j, int symbol_count=-1);	// also b_denom
  double a_denom(int i, int j, int symbol_count=-1);
  double b_numer(int i, int j, int sym, int *symbol_array,int symbol_count=-1);
  double set_cumulative_ab_counts();		
  double set_ab_counts(int* string_array,int symbol_count);
  double reestimate();				// returns sum of changes
  double test(int* string, int symbol_count=-1);// test one string 

 public:
  HMM(int symbols, int states, int new_seed=-1);// initialize random model
  HMM(char* filename);                       	// initialize pre-built model
  ~HMM();                                    	// deallocate arrays
  void batch_train(char* filename, double min_delta_psum); // train parameters
  double batch_test(char* filename);	     	// test using file
  void show_probs();                         	// display model probabilities
  void dump_model(char* filename);           	// dumps the current model
  void dump_seq(ofstream &seq, int length);     // writes one sequence 
  void set_seed(int s) { seed = s;}            	// set seed
};
