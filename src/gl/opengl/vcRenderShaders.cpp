#include "gl/vcRenderShaders.h"
#include "udPlatform/udPlatformUtil.h"

#if UDPLATFORM_IOS || UDPLATFORM_IOS_SIMULATOR
# define FRAG_HEADER "#version 300 es\nprecision highp float;\n"
# define VERT_HEADER "#version 300 es\n"
#else
# define FRAG_HEADER "#version 330 core\n#extension GL_ARB_explicit_attrib_location : enable\n"
# define VERT_HEADER "#version 330 core\n#extension GL_ARB_explicit_attrib_location : enable\n"
#endif

const char* const g_udFragmentShader = FRAG_HEADER R"shader(
uniform sampler2D u_texture;
uniform sampler2D u_depth;

layout (std140) uniform u_params
{
  vec4 u_screenParams;  // sampleStepX, sampleStepSizeY, near plane, far plane
  mat4 u_inverseViewProjection;

  // outlining
  vec4 u_outlineColour;
  vec4 u_outlineParams;   // outlineWidth, edge threshold, (unused), (unused)

  // colour by height
  vec4 u_colourizeHeightColourMin;
  vec4 u_colourizeHeightColourMax;
  vec4 u_colourizeHeightParams; // min world height, max world height, (unused), (unused)

  // colour by depth
  vec4 u_colourizeDepthColour;
  vec4 u_colourizeDepthParams; // min distance, max distance, (unused), (unused)

  // contours
  vec4 u_contourColour;
  vec4 u_contourParams; // contour distance, contour band height, (unused), (unused)
};

//Input Format
in vec2 v_texCoord;

//Output Format
out vec4 out_Colour;

float linearizeDepth(float depth)
{
  float nearPlane = u_screenParams.z;
  float farPlane = u_screenParams.w;
  return (2.0 * nearPlane) / (farPlane + nearPlane - depth * (farPlane - nearPlane));
}

float getNormalizedPosition(float v, float min, float max)
{
  return clamp((v - min) / (max - min), 0.0, 1.0);
}

// depth is packed into .w component
vec4 edgeHighlight(vec3 col, vec2 uv, float depth)
{
  vec3 sampleOffsets = vec3(u_screenParams.xy, 0.0);
  float edgeOutlineThreshold = u_outlineParams.y;
  float farPlane = u_screenParams.w;

  float d1 = texture(u_depth, uv + sampleOffsets.xz).x;
  float d2 = texture(u_depth, uv - sampleOffsets.xz).x;
  float d3 = texture(u_depth, uv + sampleOffsets.zy).x;
  float d4 = texture(u_depth, uv - sampleOffsets.zy).x;

  float wd0 = linearizeDepth(depth) * farPlane;
  float wd1 = linearizeDepth(d1) * farPlane;
  float wd2 = linearizeDepth(d2) * farPlane;
  float wd3 = linearizeDepth(d3) * farPlane;
  float wd4 = linearizeDepth(d4) * farPlane;

  float isEdge = 1.0 - step(wd0 - wd1, edgeOutlineThreshold) * step(wd0 - wd2, edgeOutlineThreshold) * step(wd0 - wd3, edgeOutlineThreshold) * step(wd0 - wd4, edgeOutlineThreshold);

  vec3 edgeColour = mix(col.xyz, u_outlineColour.xyz, u_outlineColour.w);
  float minDepth = min(min(min(d1, d2), d3), d4);
  return vec4(mix(col.xyz, edgeColour, isEdge), mix(depth, minDepth, isEdge));
}

vec3 contourColour(vec3 col, vec3 fragWorldPosition)
{
  float contourDistance = u_contourParams.x;
  float contourBandHeight = u_contourParams.y;

  float isCountour = step(contourBandHeight, mod(fragWorldPosition.z, contourDistance));
  vec3 contourColour = mix(col.xyz, u_contourColour.xyz, u_contourColour.w);
  return mix(contourColour, col.xyz, isCountour);
}

vec3 colourizeByHeight(vec3 col, vec3 fragWorldPosition)
{
  vec2 worldColourMinMax = u_colourizeHeightParams.xy;

  float minMaxColourStrength = getNormalizedPosition(fragWorldPosition.z, worldColourMinMax.x, worldColourMinMax.y);

  vec3 minColour = mix(col.xyz, u_colourizeHeightColourMin.xyz, u_colourizeHeightColourMin.w);
  vec3 maxColour = mix( col.xyz, u_colourizeHeightColourMax.xyz,u_colourizeHeightColourMax.w);
  return mix(minColour, maxColour, minMaxColourStrength);
}

vec3 colourizeByDepth(vec3 col, float depth)
{
  float farPlane = u_screenParams.w;
  float linearDepth = linearizeDepth(depth) * farPlane;
  vec2 depthColourMinMax = u_colourizeDepthParams.xy;

  float depthColourStrength = getNormalizedPosition(linearDepth, depthColourMinMax.x, depthColourMinMax.y);
  return mix(col.xyz, u_colourizeDepthColour.xyz, depthColourStrength * u_colourizeDepthColour.w);
}

void main()
{
  vec4 col = texture(u_texture, v_texCoord);
  float depth = texture(u_depth, v_texCoord).x;

  vec4 fragWorldPosition = u_inverseViewProjection * vec4(vec2(v_texCoord.x, 1.0 - v_texCoord.y) * vec2(2.0) - vec2(1.0), depth * 2.0 - 1.0, 1.0);
  fragWorldPosition /= fragWorldPosition.w;

  col.xyz = colourizeByHeight(col.xyz, fragWorldPosition.xyz);
  col.xyz = colourizeByDepth(col.xyz, depth);

  float edgeOutlineWidth = u_outlineParams.x;
  if (edgeOutlineWidth > 0.0 && u_outlineColour.w > 0.0)
  {
    vec4 edgeResult = edgeHighlight(col.xyz, v_texCoord, depth);
    col.xyz = edgeResult.xyz;
    depth = edgeResult.w; // to preserve outsides edges, depth written may be adjusted
  }
  col.xyz = contourColour(col.xyz, fragWorldPosition.xyz);

  out_Colour = vec4(col.xyz, 1.0); // UD always opaque
  gl_FragDepth = depth;
}
)shader";

const char* const g_udVertexShader = VERT_HEADER R"shader(
//Input format
layout(location = 0) in vec2 a_position;
layout(location = 1) in vec2 a_texCoord;

//Output Format
out vec2 v_texCoord;

void main()
{
  gl_Position = vec4(a_position.x, a_position.y, 0.0, 1.0);
  v_texCoord = a_texCoord;
}
)shader";

const char* const g_tileFragmentShader = FRAG_HEADER R"shader(
//Input Format
in vec4 v_colour;
in vec2 v_uv;

//Output Format
out vec4 out_Colour;

uniform sampler2D u_texture;
//uniform sampler2D u_dem;

void main()
{
  //vec4 col = texture(u_texture, v_uv);
  //out_Colour = vec4(col.xyz * v_colour.xyz, 1.0) * v_colour.w;

  //vec4 dem = texture(u_dem, v_uv);
  //out_Colour = vec4(vec3(dem.x / 10.0), 1.0);

  float t = v_uv.x / 1000.0;//(max(0.5, v_uv.x / 1000.0) - 0.5) * 2;
  out_Colour = vec4((t + v_colour.w) * v_colour.xyz, 1);
 // out_Colour = vec4(v_colour.xyz, 1);
}
)shader";

const char* const g_tileVertexShader = VERT_HEADER R"shader(
//Input format
layout(location = 0) in vec2 a_uv;

//Output Format
out vec4 v_colour;
out vec2 v_uv;

uniform sampler2D u_dem0;
uniform sampler2D u_dem1;

#define VERTEX_COUNT 3

layout (std140) uniform u_EveryObject
{
  // TODO put into u_everything
  mat4 u_projection;
  mat4 u_view;

  vec4 u_eyePositions[VERTEX_COUNT * VERTEX_COUNT];
  vec4 u_colour;
  vec4 u_demUVs[2 * VERTEX_COUNT * VERTEX_COUNT];
};

void main()
{
  vec4 eyePos = vec4(0.0);
  vec2 demUV0 = vec2(0.0);
  vec2 demUV1 = vec2(0.0);

  {
    vec2 indexUV = a_uv * 2;

    float ui = floor(indexUV.x);
    float vi = floor(indexUV.y);
    float ui2 = min(2.0, ui + 1.0);
    float vi2 = min(2.0, vi + 1.0);
    vec2 uvt = vec2(indexUV.x - ui, indexUV.y - vi);

    // bilinear position
    vec4 p0 = u_eyePositions[int(vi * 3.0 + ui)];
    vec4 p1 = u_eyePositions[int(vi * 3.0 + ui2)];
    vec4 p2 = u_eyePositions[int(vi2 * 3.0 + ui)];
    vec4 p3 = u_eyePositions[int(vi2 * 3.0 + ui2)];

    vec4 pu = (p0 + (p1 - p0) * uvt.x);
    vec4 pv = (p2 + (p3 - p2) * uvt.x);
    eyePos = (pu + (pv - pu) * uvt.y);

    // bilinear DEM heights 0
    vec2 duv0 = u_demUVs[int(vi * 3.0 + ui)].xy;
    vec2 duv1 = u_demUVs[int(vi * 3.0 + ui2)].xy;
    vec2 duv2 = u_demUVs[int(vi2 * 3.0 + ui)].xy;
    vec2 duv3 = u_demUVs[int(vi2 * 3.0 + ui2)].xy;

    vec2 duvu = (duv0 + (duv1 - duv0) * uvt.x);
    vec2 duvd = (duv2 + (duv3 - duv2) * uvt.x);
    demUV0 = (duvu + (duvd - duvu) * uvt.y);

    // bilinear DEM heights 1
    duv0 = u_demUVs[9 + int(vi * 3.0 + ui)].xy;
    duv1 = u_demUVs[9 + int(vi * 3.0 + ui2)].xy;
    duv2 = u_demUVs[9 + int(vi2 * 3.0 + ui)].xy;
    duv3 = u_demUVs[9 + int(vi2 * 3.0 + ui2)].xy;

    duvu = (duv0 + (duv1 - duv0) * uvt.x);
    duvd = (duv2 + (duv3 - duv2) * uvt.x);
    demUV1 = (duvu + (duvd - duvu) * uvt.y);
  }

  float tileHeight = 0.0;

  float use0 = float(demUV0.x >= 0.0 && demUV0.x <= 1.0 && demUV0.y >= 0.0 && demUV0.y <= 1.0);
  float use1 = float(demUV1.x >= 0.0 && demUV1.x <= 1.0 && demUV1.y >= 0.0 && demUV1.y <= 1.0);
  if (use0 == 0.0 && use1 == 0.0)
  {

  } else if (use0 == 0.0)
  {
    tileHeight = texture(u_dem1, demUV1).r;
  } else
  {
    tileHeight = texture(u_dem0, demUV0).r;
  }
  tileHeight *= 65536.0;

  vec4 h = u_view * vec4(0, 0, tileHeight, 1.0);
  vec4 baseH = u_view * vec4(0, 0, 0, 1.0);
  vec4 diff = h - baseH;
  vec4 finalClipPos = u_projection * (eyePos + diff);

  v_uv = vec2(tileHeight);//a_uv.xy;
  v_colour = u_colour;
  if (use0 == 0.0)
    v_colour = vec4(1,0,0,u_colour.w);
  else
    v_colour = vec4(0,1,0,u_colour.w);

  gl_Position = finalClipPos;

}
)shader";


const char* const g_vcSkyboxVertexShader = VERT_HEADER R"shader(
//Input format
layout(location = 0) in vec2 a_position;
layout(location = 1) in vec2 a_texCoord;

//Output Format
out vec2 v_texCoord;

void main()
{
  gl_Position = vec4(a_position.x, a_position.y, 0.0, 1.0);
  v_texCoord = vec2(a_texCoord.x, 1.0 - a_texCoord.y);
}
)shader";

const char* const g_vcSkyboxFragmentShader = FRAG_HEADER R"shader(

uniform sampler2D u_texture;
layout (std140) uniform u_EveryFrame
{
  mat4 u_inverseViewProjection;
};

//Input Format
in vec2 v_texCoord;

//Output Format
out vec4 out_Colour;

#define PI 3.14159265359

vec2 directionToLatLong(vec3 dir)
{
  vec2 longlat = vec2(atan(dir.x, dir.y) + PI, acos(dir.z));
  return longlat / vec2(2.0 * PI, PI);
}

void main()
{
  // work out 3D point
  vec4 point3D = u_inverseViewProjection * vec4(v_texCoord * vec2(2.0) - vec2(1.0), 1.0, 1.0);
  point3D.xyz = normalize(point3D.xyz / point3D.w);
  vec4 c1 = texture(u_texture, directionToLatLong(point3D.xyz));

  out_Colour = c1;
}
)shader";


const char* const g_PositionNormalFragmentShader = FRAG_HEADER R"shader(
  //Input Format
  in vec4 v_colour;
  in vec3 v_normal;
  in vec4 v_fragClipPosition;
  in vec3 v_sunDirection;

  //Output Format
  out vec4 out_Colour;

  void main()
  {
    // fake a reflection vector
    vec3 fakeEyeVector = normalize(v_fragClipPosition.xyz / v_fragClipPosition.w);
    vec3 worldNormal = normalize(v_normal * vec3(2.0) - vec3(1.0));
    float ndotl = 0.5 + 0.5 * -dot(v_sunDirection, worldNormal);
    float edotr = max(0.0, -dot(-fakeEyeVector, worldNormal));
    edotr = pow(edotr, 60.0);
    vec3 sheenColour = vec3(1.0, 1.0, 0.9);
    out_Colour = vec4(v_colour.a * (ndotl * v_colour.xyz + edotr * sheenColour), 1.0);
  }
)shader";

const char* const g_PositionNormalVertexShader = VERT_HEADER R"shader(
  //Input Format
  layout(location = 0) in vec3 a_pos;
  layout(location = 1) in vec3 a_normal;

  //Output Format
  out vec4 v_colour;
  out vec3 v_normal;
  out vec4 v_fragClipPosition;
  out vec3 v_sunDirection;

  layout (std140) uniform u_EveryObject
  {
    mat4 u_viewProjectionMatrix;
    vec4 u_colour;
    vec3 u_sunDirection;
    float _padding;
  };

  void main()
  {
    gl_Position = u_viewProjectionMatrix * vec4(a_pos, 1.0);

    v_normal = ((a_normal * 0.5) + 0.5);
    v_colour = u_colour;
    v_sunDirection = u_sunDirection;
    v_fragClipPosition = gl_Position;
  }
)shader";

const char* const g_ImGuiVertexShader = VERT_HEADER R"shader(
layout (std140) uniform u_EveryFrame
{
  mat4 ProjMtx;
};

layout(location = 0) in vec2 Position;
layout(location = 1) in vec2 UV;
layout(location = 2) in vec4 Color;

out vec2 Frag_UV;
out vec4 Frag_Color;

void main()
{
  Frag_UV = UV;
  Frag_Color = Color;
  gl_Position = ProjMtx * vec4(Position.xy, 0, 1);
}
)shader";

const char* const g_ImGuiFragmentShader = FRAG_HEADER R"shader(

uniform sampler2D Texture;

in vec2 Frag_UV;
in vec4 Frag_Color;

out vec4 Out_Color;

void main()
{
  Out_Color = Frag_Color * texture(Texture, Frag_UV.st);
}
)shader";

const char* const g_FenceVertexShader = VERT_HEADER R"shader(
layout (std140) uniform u_EveryFrame
{
  vec4 u_bottomColour;
  vec4 u_topColour;

  float u_orientation;
  float u_width;
  float u_textureRepeatScale;
  float u_textureScrollSpeed;
  float u_time;

  vec3 _padding;
};

layout (std140) uniform u_EveryObject
{
  mat4 u_modelViewProjectionMatrix;
};

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec4 a_ribbonInfo; // xyz: expand vector; z: pair id (0 or 1)

out vec2 v_uv;
out vec4 v_colour;

void main()
{
  // fence horizontal UV pos packed into Y channel
  v_uv = vec2(mix(a_uv.y, a_uv.x, u_orientation) * u_textureRepeatScale - u_time * u_textureScrollSpeed, a_ribbonInfo.w);
  v_colour = mix(u_bottomColour, u_topColour, a_ribbonInfo.w);

  // fence or flat
  vec3 worldPosition = a_position + mix(vec3(0, 0, a_ribbonInfo.w) * u_width, a_ribbonInfo.xyz, u_orientation);

  gl_Position = u_modelViewProjectionMatrix * vec4(worldPosition, 1.0);
}
)shader";

const char* const g_FenceFragmentShader = FRAG_HEADER R"shader(
  //Input Format
  in vec2 v_uv;
  in vec4 v_colour;

  //Output Format
  out vec4 out_Colour;

  uniform sampler2D u_texture;

  void main()
  {
    vec4 texCol = texture(u_texture, v_uv);
    out_Colour = vec4(texCol.xyz * v_colour.xyz, 1.0) * texCol.w * v_colour.w;
  }
)shader";
