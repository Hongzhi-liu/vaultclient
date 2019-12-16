#ifndef vcShaderCondunctor__h_
#define vcShaderCondunctor__h_

#include <ShaderConductor.hpp>
using namespace ShaderConductor;

#include "udResult.h"

bool vcShaderConductor_BuildAll();
udResult vcShaderConductor_Compile(const char *pInput, const char *pEntry, ShaderStage stage, const char *pOutput);

#endif //vcShaderCondunctor__h_
