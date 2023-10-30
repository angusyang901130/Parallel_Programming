#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //

  __pp_vec_float x;
  __pp_vec_int y;
  __pp_vec_float result;
  __pp_vec_int cnt;

  __pp_vec_int zero_int = _pp_vset_int(0);
  __pp_vec_int one_int = _pp_vset_int(1);

  __pp_vec_float zero_float = _pp_vset_float(0.f);
  __pp_vec_float one_float = _pp_vset_float(1.f);

  __pp_vec_float clamp = _pp_vset_float(9.999999f);

  __pp_mask maskAll, maskIsZero, maskIsNotZero, maskNeedClamp;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  { 
    if(i + VECTOR_WIDTH > N)
      i = N - VECTOR_WIDTH;
    
    // All Ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsZero = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll);    // x = values[i];

    // Load vector of exponents from contiguous memory addresses
    _pp_vload_int(y, exponents + i, maskAll);   // y = exponents[i];

    // Set mask according to predicate
    _pp_veq_int(maskIsZero, y, zero_int, maskAll);    // if (y == 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vadd_float(result, one_float, zero_float, maskIsZero);    // output[i] = 1.f;

    // Inverse maskIsZero to generate "else" mask
    maskIsNotZero = _pp_mask_not(maskIsZero);   // } else {

    // Initialize result with x if exponents are not zero
    _pp_vadd_float(result, x, zero_float, maskIsNotZero);   // result = x;

    // Count assignment
    _pp_vsub_int(cnt, y, one_int, maskIsNotZero);   // count = y - 1;

    // Update maskIsNotZero
    _pp_vgt_int(maskIsNotZero, cnt, zero_int, maskIsNotZero);   // while (count > 0) {

    // Check whether multiplication is still needed
    while(_pp_cntbits(maskIsNotZero)){    
      _pp_vmult_float(result, result, x, maskIsNotZero);    // result *= x;
      _pp_vsub_int(cnt, cnt, one_int, maskIsNotZero);       // count --;
      _pp_vgt_int(maskIsNotZero, cnt, zero_int, maskIsNotZero);   
    }                                                           // }

    // Check if result need clamp
    _pp_vgt_float(maskNeedClamp, result, clamp, maskIsNotZero);   // if (result > 9.999999f) {

    // Assign result to 9.999999f if needed clamp
    _pp_vadd_float(result, clamp, zero_float, maskNeedClamp);     // result = 9.999999f;  }

    // Write result back to memory
    _pp_vstore_float(output, result, maskAll);

  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  float sum = 0;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_vec_float x;

  __pp_mask maskAll = _pp_init_ones();

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    _pp_vload_float(x, values + i, maskAll);

    _pp_hadd_float(x, x);

  }

  return 0.0;
}