#include <wb.h>
#include <amp.h>

using namespace concurrency;

int main(int argc, char **argv)
{
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;

  args = wbArg_read(argc, argv);

  hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  array_view<float, 1> deviceInput1(inputLength, hostInput1);
  array_view<float, 1> deviceInput2(inputLength, hostInput2);
  array_view<float, 1> deviceOutput(inputLength, hostOutput);
  
  deviceOutput.discard_data(); // the data will be overwritten
  
  parallel_for_each(deviceOutput.get_extent(),
                    [=](index<1> i) restrict(amp)
  {
    deviceOutput[i] = deviceInput1[i] + deviceInput2[i];
  });
  deviceOutput.synchronize();

  // Compare the solutions
  wbSolution(args, hostOutput, inputLength);

  return 0;
}

