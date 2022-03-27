# makefile for hidden Markov model code.
# modified by Arthur Stephens to add Solaris options

## CC: uncomment the following for Solaris

# CFLAGS = -c -O
# CC = CC
# GLIB =
# SOLLIB = -L/usr/ucblib -lucb

# the next three lines perform purify

# PURIFY = /opt/purify3/purify-3.0-solaris2/purify
# PURIFY_FLAGS = -best-effort -cache_dir=/tmp 
# PURE = $(PURIFY) $(PURIFY_FLAGS)

## CC: end 

## gcc: uncomment the following for gcc

CFLAGS = -c -O -D_random_h_  # ignore declarations in random.h
CC = g++
#GLIB = 
SOLLIB =
PURE =

## gcc: end

all : train_hmm test_hmm generate_seq

train_hmm : hmm.o train_hmm.o
	$(PURE) $(CC) -o train_hmm hmm.o train_hmm.o -lm $(SOLLIB) $(GLIB)

test_hmm : hmm.o test_hmm.o
	$(CC) -o test_hmm hmm.o test_hmm.o -lm $(SOLLIB) $(GLIB)

generate_seq : hmm.o generate_seq.o
	$(CC) -o generate_seq generate_seq.o hmm.o -lm $(SOLLIB) $(GLIB)

train_hmm.o : train_hmm.cc hmm.h
	$(CC) $(CFLAGS) train_hmm.cc

test_hmm.o : test_hmm.cc hmm.h
	$(CC) $(CFLAGS) test_hmm.cc

generate_seq.o : generate_seq.cc hmm.h
	$(CC) $(CFLAGS) generate_seq.cc 

hmm.o : hmm.cc hmm.h 
	$(CC) $(CFLAGS) hmm.cc 

clean : 
	rm -f *.o core *.seq *.seq.hmm
