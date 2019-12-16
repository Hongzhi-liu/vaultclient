#ifndef vcRenderShaders_h__
#define vcRenderShaders_h__

#include "vcGLState.h"

enum ShaderIndexes
{
  vcSI_Visualization = 0,
  vcSI_PostEffects = 1,
  vcSI_ViewShed = 2,
  vcSI_UD = 3,
  vcSI_Splat = 4,
  vcSI_Tile = 5,
  vcSI_SkyboxPanorama = 6,
  vcSI_SkyboxImageColour = 7,
  vcSI_Fence = 8,
  vcSI_Water = 9,
  vcSI_Compass = 10,
  vcSI_ImGui = 11,
  vcSI_PolygonP3N3UV2 = 12,
  vcSI_FlatColour = 13,
  vcSI_DepthOnly = 14,
  vcSI_ImageRenderer = 15,
  vcSI_ImageRendererBillboard = 16,
  vcSI_Blur = 17,
  vcSI_Highlight = 18,
  vcSI_Count = 19
};

extern const char *g_VertexShaders[];
extern const char *g_FragmentShaders[];

// GPU UD Renderer
extern const char *g_udGPURenderQuadVertexShader;
extern const char *g_udGPURenderQuadFragmentShader;
extern const char *g_udGPURenderGeomVertexShader;
extern const char *g_udGPURenderGeomFragmentShader;
extern const char *g_udGPURenderGeomGeometryShader;

#endif//vcRenderShaders_h__
