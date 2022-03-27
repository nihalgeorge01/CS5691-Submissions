//   file : random.h
// version: 1.03 [August 21, 1995]

#ifndef _random_h_
#define _random_h_

/* Mods by Art Stephens so compiles on Sun using CC, does anyone know */
/* the correct header files to use? */
extern "C" {
  long random(void);
  int srandom(unsigned int seed);
};

#endif // _random_h_
