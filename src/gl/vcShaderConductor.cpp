/*
 * ShaderConductor
 *
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this
 * software and associated documentation files (the "Software"), to deal in the Software
 * without restriction, including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
 * to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
 * FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

// Substantial portions of this code are copied with few modifications from the ShaderConductor wrapper/command line tool

#include "vcShaderConductor.h"
#include "udStringUtil.h"
#include "udFile.h"
#include "vcGLState.h"
#include "vcState.h"
#include "vcRenderShaders.h"

const char *g_VertexShaders[vcSI_Count] = {};
const char *g_FragmentShaders[vcSI_Count] = {};

udResult vcShaderConductor_Compile(const char *pInput, const char *pEntry, ShaderStage stage, const char **ppOutput)
{
  if (pInput == nullptr)
    return udR_InvalidParameter_;

#if GRAPHICS_API_D3D11
  udUnused(stage);
  udUnused(ppOutput);
  return udR_Success;
#else
  udResult result;

  Compiler::SourceDesc desc;
  desc.source = pInput;
  desc.fileName = "";
  desc.entryPoint = pEntry;
  desc.stage = stage;
  desc.defines = nullptr;
  desc.numDefines = 0;

  Compiler::TargetDesc targ;
  targ.version = nullptr;
#if UDPLATFORM_IOS || UDPLATFORM_IOS_SIMULATOR
  targ.language = ShadingLanguage::Msl_iOS;
#elif GRAPHICS_API_METAL
  targ.language = ShadingLanguage::Msl_macOS;
#elif GRAPHICS_API_OPENGL
  targ.language = ShadingLanguage::Glsl;
  targ.version = "330";
#endif

  const char *extMap[] = { "dxil", "spv", "hlsl", "glsl", "essl", "msl", "msl" };

  Compiler::Options opts = {};
  opts.packMatricesInRowMajor = false;

  Compiler::ResultDesc compileResult = Compiler::Compile(desc, opts, targ);

  if (compileResult.errorWarningMsg != nullptr)
  {
    const char *msg = reinterpret_cast<const char *>(compileResult.errorWarningMsg->Data());
    UDASSERT(0, udTempStr("Error or warning form shader compiler: %s", msg));
    UD_ERROR_SET(udR_Failure_);
  }

  UD_ERROR_NULL(compileResult.target, udR_Failure_);
  *ppOutput = udStrndup((const char *)compileResult.target->Data(), compileResult.target->Size());

  static int index = 0;
  udFile_Save(udTempStr("asset://shader-%d.glsl", index++), *ppOutput, compileResult.target->Size());

  result = udR_Success;

epilogue:
  DestroyBlob(compileResult.errorWarningMsg);
  DestroyBlob(compileResult.target);

  return result;
#endif
}

bool vcShaderConductor_BuildAll()
{
  char *pSource = nullptr;
  udFile_Load("asset://vcRenderShaders.hlsl", &pSource);

  char *pScan = pSource, *pPos;

  int depth = 0;

  const char **pShaders = g_VertexShaders;
  ShaderStage stage = ShaderStage::VertexShader;

  for (int j = 0; j < 2; ++j)
  {
    if (j == 1)
    {
      pShaders = g_FragmentShaders;
      stage = ShaderStage::PixelShader;
    }

    for (int i = 0; i < vcSI_Count; ++i)
    {
      while (*pScan != '{')
        ++pScan;

      pPos = pScan;

      while (*(++pScan) != '}' || depth)
      {
        if (*pScan == '{')
          ++depth;
        else if (*pScan == '}')
          --depth;
      }

      *pScan = '\0';

      if (*(++pPos) != '\0')
      {
        if (vcShaderConductor_Compile(pPos, "main", stage, &pShaders[i]) != udR_Success)
          return false;
      }
    }
  }

  return true;
}
