//   file : train_hmm.cc
// version: 1.03 [August 21, 1995]
//
/*
  Copyright (C) 1994 Richard Myers and James Whitson
  Permission is granted to any individual or institution to use, copy, or
  redistribute this software so long as all of the original files are 
  included unmodified, that it is not sold for profit, and that this 
  copyright notice is retained.
*/
// This program creates a hidden markov model from a series of symbol sequences
// using an initial predefined model or randomly generated model.

using namespace std;

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include "hmm.h"

main(int argc, char* argv[])

{
  // get command line parms

  char* train_file=argv[1];
  char* initial_model;
  double min_delta_psum;
  int states,symbols,seed;

  // check args
  if (argc != 4 && argc != 6) {
    cerr << "ERROR: Too few arguments.\n\n" << 
    "Usage: "<< argv[0]<<" <train_file> <hmm_model> <min_delta_psum> or \n" <<
    "       "<< argv[0]<<" <train_file> <seed> <nstates> <nsymbols> <min_delta_psum>\n\n";
    exit (-1);
  }
  
  if (argc==4) {
    initial_model=argv[2];
    min_delta_psum=atof(argv[3]);
  }

  if (argc==6) {
    seed = atoi(argv[2]);
    states = atoi(argv[3]);
    symbols = atoi(argv[4]);
    min_delta_psum=atof(argv[5]);
  }

  HMM *hmm;
  if (argc==4) {
    // initialize model from model file
    hmm = new HMM(initial_model);
    // train model on data
    hmm->batch_train(train_file,min_delta_psum);
  }
  else {
    // initialize model randomly
    hmm = new HMM(symbols, states, seed);
    // train model on data
    hmm->batch_train(train_file,min_delta_psum);
  }
  
  // dump the resulting model to a file
  char newfilename[100];
  sprintf(newfilename,"%s.hmm",train_file);
  hmm->dump_model(newfilename);
  
  delete hmm;
}

