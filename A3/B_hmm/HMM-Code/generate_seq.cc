//   file : generate_seq.cc
//  author: Richard Myers
// version: 1.03 [August 21, 1995]
//
// This program generates random sequences using the probabilities defined
// in a markov model file.

using namespace std;

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include "hmm.h"
#include "random.h"

main(int argc, char* argv[])
{
  // check args
  if (argc < 4) {
    cerr << "ERROR: Too few arguments.\n\n" <<
    "Usage: " << argv[0] << " <hmm_model> <num_seq> <min_seq_len> [ <max_seq_len> <seed> ]\n\n";
    exit (-1);
  }
  
  // get command line parms
  char* hmm_def_file = argv[1];
  int num_seq = atoi(argv[2]);
  int min_seq_len = atoi(argv[3]);
  int max_seq_len;
  int seq_len, var;
  double len_prob;

  // initialize using parameters

  if ( argc < 5 )
    max_seq_len = min_seq_len; // fixed length strings
  else
    max_seq_len = atoi(argv[4]);

  if ( argc > 5 )
    srandom(atoi(argv[5]));  

  var = max_seq_len - min_seq_len;

  // create the hmm from model file
  HMM *hmm = new HMM(hmm_def_file);
  
  // dump the resulting model to a file
  char filename[100];
  sprintf(filename,"%s.seq",hmm_def_file);

  ofstream seq(filename);
  if (!seq) {
    cerr << "ERROR: Couldn't create file for sequence dump.";
    exit(-1);
  }

  if ( var > 0 )
    for (int i=0; i <  num_seq; i++) {
      seq_len = min_seq_len + (random() % var);
      hmm->dump_seq( seq, seq_len );
    }
  else
    for (int i=0; i <  num_seq; i++) {
      hmm->dump_seq( seq, min_seq_len );
    }

  cout << "\nDumped sequences to file ==> " << filename << "\n";
  seq.close();
}


