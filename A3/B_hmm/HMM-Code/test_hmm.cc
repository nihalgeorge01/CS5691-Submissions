//   file : hmm.cc
// authors: Richard Myers and Jim Whitson
// version: 1.03 [August 21, 1995]

using namespace std;

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include "hmm.h"

main(int argc, char* argv[])
{
  // check args
  if (argc < 3) {
    cerr << "ERROR: Too few arguments.\n\n" << 
    "Usage: "<< argv[0]<<" <test_file> <initial_model>\n";
    exit (-1);
  }
  
  // get command line parms
  char* train_file=argv[1];
  char* initial_model=argv[2];

  // initialize model from model file
  HMM *hmm = new HMM(initial_model);
  // test model on data
  hmm->batch_test(train_file);
}

