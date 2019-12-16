#ifndef vcShader_h__
#define vcShader_h__

#include "udMath.h"
#include "gl/vcLayout.h"

struct vcShader;
struct vcShaderSampler;
struct vcShaderConstantBuffer;
struct vcTexture;

bool vcShader_CreateFromText(vcShader **ppShader, const char *pVertexShader, const char *pFragmentShader, const vcVertexLayoutTypes *pInputTypes, uint32_t totalInputs, const char *pGeometryShader = nullptr);
template <size_t N> inline bool vcShader_CreateFromText(vcShader **ppShader, const char *pVertexShader, const char *pFragmentShader, const vcVertexLayoutTypes (&inputTypes)[N], const char *pGeometryShader = nullptr) { return vcShader_CreateFromText(ppShader, pVertexShader, pFragmentShader, inputTypes, (uint32_t)N, pGeometryShader); }
void vcShader_DestroyShader(vcShader **ppShader);

bool vcShader_Bind(vcShader *pShader); // nullptr to unbind shader

bool vcShader_BindTexture(vcShader *pShader, vcTexture *pTexture, uint16_t samplerIndex, vcShaderSampler *pSampler = nullptr);

bool vcShader_GetConstantBuffer(vcShaderConstantBuffer **ppBuffer, vcShader *pShader, const char *pBufferName, const size_t bufferSize);
bool vcShader_BindConstantBuffer(vcShader *pShader, vcShaderConstantBuffer *pBuffer, const void *pData, const size_t bufferSize);
bool vcShader_ReleaseConstantBuffer(vcShader *pShader, vcShaderConstantBuffer *pBuffer);

bool vcShader_GetSamplerIndex(vcShaderSampler **ppSampler, vcShader *pShader, const char *pSamplerName);

// ShaderConductor prepends type_ to buffer names regardless of what name you choose
#if GRAPHICS_API_OPENGL
static const char *g_EveryObject = "type_u_EveryObject";
static const char *g_EveryFrameVert = "type_u_EveryFrameVert";
static const char *g_EveryFrameFrag = "type_u_EveryFrameFrag";
static const char *g_VertParams = "type_u_VertParams";
static const char *g_FragParams = "type_u_FragParams";

static const char *g_ColourSampler = "SPIRV_Cross_CombinedcolourTexturecolourSampler";
static const char *g_DepthSampler = "SPIRV_Cross_CombineddepthTexturedepthSampler";

#else
static const char *g_EveryObject = "u_EveryObject";
static const char *g_EveryFrameVert = "u_EveryFrameVert";
static const char *g_EveryFrameFrag = "u_EveryFrameFrag";
static const char *g_VertParams = "u_VertParams";
static const char *g_FragParams = "u_FragParams";

static const char *g_ColourSampler = "colourSampler";
static const char *g_DepthSampler = "depthSampler";
#endif

#endif
