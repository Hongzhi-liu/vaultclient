#include "vcSceneLayer.h"

#include "vcSettings.h"

#include "gl/vcLayout.h"

#include "udGeoZone.h"
#include "udJSON.h"

struct vcPolygonModel;
struct vcTexture;

enum
{
  // TODO: (EVC-541) Determine actual max URL length of I3S
  vcMaxURLLength = vcMaxPathLength,
};

enum vcSceneLayerLoadType
{
  vcSLLT_None,

  vcSLLT_Convert,
  vcSLLT_Rendering
};

struct vcSceneLayerNode
{
  enum vcLoadState
  {
    vcLS_NotLoaded,
    vcLS_Failed,
    vcLS_InQueue,
    vcLS_Loading,

    vcLS_PartialLoad, // TODO: (EVC-569) Partial node uploading has not been fully tested
    vcLS_InMemory,
    vcLS_Success,
  };

  volatile vcLoadState loadState;

  int level;
  char id[vcMaxURLLength];
  const char *pURL;
  vcSceneLayerNode *pChildren;
  size_t childrenCount;

  udDouble2 latLong; // this is taken from the MBS
  struct
  {
    udDouble3 position;
    double radius;
  } minimumBoundingSphere; // Cartesian
  //int lodSelectionMetric; // TODO
  double lodSelectionValue;
  udGeoZone zone;

  struct SharedResource
  {
    const char *pURL;

  } *pSharedResources;
  size_t sharedResourceCount;

  struct FeatureData
  {
    const char *pURL;

    udDouble3 position;
    udDouble3 pivotOffset;
    udDouble4 minimumBoundingBox;
  } *pFeatureData;
  size_t featureDataCount;

  struct GeometryData
  {
    bool loaded;
    const char *pURL;

    udDouble4x4 originMatrix;
    //udDouble4x4 transformation;
    //int type;
    vcPolygonModel *pModel;

    // cpu
    uint8_t *pData;
    uint64_t vertCount;
    size_t vertexStride;
  } *pGeometryData;
  size_t geometryDataCount;

  struct TextureData
  {
    bool loaded;
    const char *pURL;

    vcTexture *pTexture;

    // cpu
    uint8_t *pData;
    int width, height;
  } *pTextureData;
  size_t textureDataCount;

  struct AttributeData
  {
    const char *pURL;

  } *pAttributeData;
  size_t attributeDataCount;
};

struct vcSceneLayer
{
  vWorkerThreadPool *pThreadPool;
  bool isActive;

  char sceneLayerURL[vcMaxURLLength];
  char pathSeparatorChar; // '/' or '\\'
  udJSON description;
  udDouble4 extent;

  vcSceneLayerNode root;

  vcVertexLayoutTypes *pDefaultGeometryLayout;
  size_t defaultGeometryLayoutCount;

  // TODO: (EVC-553) This is temporary
  uint64_t gpuBytesUploadedThisFrame;
};

udResult vcSceneLayer_LoadNode(vcSceneLayer *pSceneLayer, vcSceneLayerNode *pNode = nullptr, const vcSceneLayerLoadType &loadType = vcSLLT_None);

// Prepares the node for use
// Returns true if the node is ready for use
bool vcSceneLayer_TouchNode(vcSceneLayer *pSceneLayer, vcSceneLayerNode *pNode);

// TODO: (EVC-540) ASSUMPTIONS! (assumed a specific vertex layout!)
struct vcSceneLayerVertex
{
  udFloat3 position;
  udFloat3 normal;
  udFloat2 uv0;
  uint32_t colour;
};
vcSceneLayerVertex* vcSceneLayer_GetVertex(vcSceneLayerNode::GeometryData *pGeometry, uint64_t index);