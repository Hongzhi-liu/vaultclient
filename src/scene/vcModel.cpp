#include "vcModel.h"

#include "vcState.h"
#include "vcRender.h"

#include "gl/vcTexture.h"

#include "imgui.h"
#include "imgui_ex/imgui_udValue.h"
#include "imgui_ex/vcImGuiSimpleWidgets.h"

#include "vdkPointCloud.h"

#include "udStringUtil.h"

struct vcModelLoadInfo
{
  vcState *pProgramState;
  vcModel *pModel;
};

void vcModel_LoadMetadata(vcState *pProgramState, vcModel *pModel)
{
  const char *pMetadata;

  if (vdkPointCloud_GetMetadata(pProgramState->pVDKContext, pModel->m_pPointCloud, &pMetadata) == vE_Success)
  {
    pModel->m_metadata.Parse(pMetadata);
    pModel->m_hasWatermark = pModel->m_metadata.Get("Watermark").IsString();

    pModel->m_meterScale = pModel->m_metadata.Get("info.meterScale").AsDouble(1.0);
    pModel->m_pivot = pModel->m_metadata.Get("info.pivot").AsDouble3();

    vcSRID srid = 0;
    const char *pSRID = pModel->m_metadata.Get("ProjectionID").AsString();
    const char *pWKT = pModel->m_metadata.Get("ProjectionWKT").AsString();

    if (pSRID != nullptr)
    {
      pSRID = udStrchr(pSRID, ":");
      if (pSRID != nullptr)
        srid = udStrAtou(&pSRID[1]);

      pModel->m_pPreferredProjection = udAllocType(udGeoZone, 1, udAF_Zero);
      if (udGeoZone_SetFromSRID(pModel->m_pPreferredProjection, srid) != udR_Success)
        udFree(pModel->m_pPreferredProjection);
    }

    if (pModel->m_pPreferredProjection == nullptr && pWKT != nullptr)
    {
      pModel->m_pPreferredProjection = udAllocType(udGeoZone, 1, udAF_Zero);

      if (udGeoZone_SetFromWKT(pModel->m_pPreferredProjection, pWKT) != udR_Success)
        udFree(pModel->m_pPreferredProjection);
    }

    // TODO: Handle this better (EVC-535)
    const char *pMinMaxIntensity = pModel->m_metadata.Get("AttrMinMax_udIntensity").AsString();
    if (pMinMaxIntensity != nullptr)
    {
      int charCount = 0;
      int minIntensity = (uint16_t)udStrAtoi(pMinMaxIntensity, &charCount);
      int maxIntensity = (uint16_t)(udStrAtoi(pMinMaxIntensity + charCount + 1));

      pProgramState->settings.visualization.minIntensity = udMax(pProgramState->settings.visualization.minIntensity, minIntensity);
      pProgramState->settings.visualization.maxIntensity = udMin(pProgramState->settings.visualization.maxIntensity, maxIntensity);
    }
  }

  vdkPointCloud_GetStoredMatrix(pProgramState->pVDKContext, pModel->m_pPointCloud, pModel->m_sceneMatrix.a);
  pModel->m_defaultMatrix = pModel->m_sceneMatrix;
  pModel->m_baseMatrix = pModel->m_defaultMatrix;

  pModel->OnNodeUpdate(pProgramState);

  if (pModel->m_pPreferredProjection)
    pModel->m_changeZones = true;
}

void vcModel_LoadModel(void *pLoadInfoPtr)
{
  vcModelLoadInfo *pLoadInfo = (vcModelLoadInfo*)pLoadInfoPtr;
  if (pLoadInfo->pProgramState->programComplete)
    return;

  int32_t status = udInterlockedCompareExchange(&pLoadInfo->pModel->m_loadStatus, vcSLS_Loading, vcSLS_Pending);

  if (status == vcSLS_Pending)
  {
    vdkError modelStatus = vdkPointCloud_Load(pLoadInfo->pProgramState->pVDKContext, &pLoadInfo->pModel->m_pPointCloud, pLoadInfo->pModel->m_pNode->pURI);

    if (modelStatus == vE_Success)
    {
      vcModel_LoadMetadata(pLoadInfo->pProgramState, pLoadInfo->pModel);
      pLoadInfo->pModel->m_loadStatus = vcSLS_Loaded;
    }
    else if (modelStatus == vE_OpenFailure)
    {
      pLoadInfo->pModel->m_loadStatus = vcSLS_OpenFailure;
    }
    else
    {
      pLoadInfo->pModel->m_loadStatus = vcSLS_Failed;
    }
  }
}

vcModel::vcModel(vdkProject *pProject, vdkProjectNode *pNode, vcState *pProgramState) :
  vcSceneItem(pProject, pNode, pProgramState),
  m_pPointCloud(nullptr),
  m_pivot(udDouble3::zero()),
  m_defaultMatrix(udDouble4x4::identity()),
  m_pCurrentZone(nullptr),
  m_sceneMatrix(udDouble4x4::identity()),
  m_baseMatrix(udDouble4x4::identity()),
  m_changeZones(false),
  m_meterScale(0.0),
  m_hasWatermark(false),
  m_pWatermark(nullptr)
{
  vcModelLoadInfo *pLoadInfo = udAllocType(vcModelLoadInfo, 1, udAF_Zero);
  if (pLoadInfo != nullptr)
  {
    // Prepare the load info
    pLoadInfo->pModel = this;
    pLoadInfo->pProgramState = pProgramState;

    // Queue for load
    udWorkerPool_AddTask(pProgramState->pWorkerPool, vcModel_LoadModel, pLoadInfo, true);
  }
  else
  {
    m_loadStatus = vcSLS_Failed;
  }
}

vcModel::vcModel(vcState *pProgramState, const char *pName, vdkPointCloud *pCloud) :
  vcSceneItem(pProgramState, "UDS", pName),
  m_pPointCloud(nullptr),
  m_pivot(udDouble3::zero()),
  m_defaultMatrix(udDouble4x4::identity()),
  m_pCurrentZone(nullptr),
  m_sceneMatrix(udDouble4x4::identity()),
  m_baseMatrix(udDouble4x4::identity()),
  m_changeZones(false),
  m_meterScale(0.0),
  m_hasWatermark(false),
  m_pWatermark(nullptr)
{
  m_pPointCloud = pCloud;
  m_loadStatus = vcSLS_Loaded;

  vcModel_LoadMetadata(pProgramState, this);

  m_pNode->pUserData = this;
}

void vcModel::AddToScene(vcState *pProgramState, vcRenderData *pRenderData)
{
  if (m_changeZones)
    ChangeProjection(pProgramState->gis.zone);

  pRenderData->models.PushBack(this);
}

void vcModel::OnNodeUpdate(vcState *pProgramState)
{
  udDouble3 *pPosition = nullptr;
  double scale;
  udDouble3 euler;

  if (m_pNode->geomCount != 0)
  {
    if (m_pCurrentZone != nullptr)
    {
      vcProject_FetchNodeGeometryAsCartesian(pProgramState->activeProject.pProject, m_pNode, *m_pCurrentZone, &pPosition, nullptr);
    }
    else if (m_pPreferredProjection != nullptr)
    {
      vcProject_FetchNodeGeometryAsCartesian(pProgramState->activeProject.pProject, m_pNode, *m_pPreferredProjection, &pPosition, nullptr);
      m_pCurrentZone = (udGeoZone*)udMemDup(m_pPreferredProjection, sizeof(udGeoZone), 0, udAF_None);
    }
    else
    {
      vcProject_FetchNodeGeometryAsCartesian(pProgramState->activeProject.pProject, m_pNode, pProgramState->gis.zone, &pPosition, nullptr);

      if (pProgramState->gis.isProjected)
        m_pCurrentZone = (udGeoZone*)udMemDup(&pProgramState->gis.zone, sizeof(udGeoZone), 0, udAF_None);
    }

    vdkProjectNode_GetMetadataDouble(m_pNode, "transform.rotation.y", &euler.x, 0.0);
    vdkProjectNode_GetMetadataDouble(m_pNode, "transform.rotation.p", &euler.y, 0.0);
    vdkProjectNode_GetMetadataDouble(m_pNode, "transform.rotation.r", &euler.z, 0.0);
    vdkProjectNode_GetMetadataDouble(m_pNode, "transform.scale", &scale, udMag3(m_baseMatrix.axis.x));

    m_sceneMatrix = udDouble4x4::translation(m_pivot) * udDouble4x4::rotationYPR(UD_DEG2RAD(euler), *pPosition) * udDouble4x4::scaleUniform(scale) * udDouble4x4::translation(-m_pivot);
  }

  udFree(pPosition);

  m_changeZones = true;
}

void vcModel::ChangeProjection(const udGeoZone &newZone)
{
  if (m_loadStatus != vcSLS_Loaded)
  {
    m_changeZones = true;
    return;
  }

  m_changeZones = false;

  if (newZone.srid == 0)
    return;

  if (m_pCurrentZone != nullptr)
    m_sceneMatrix = udGeoZone_TransformMatrix(m_sceneMatrix, *m_pCurrentZone, newZone);
  else if (m_pPreferredProjection != nullptr)
    m_sceneMatrix = udGeoZone_TransformMatrix(m_baseMatrix, *m_pPreferredProjection, newZone);

  if (m_pCurrentZone == nullptr)
    m_pCurrentZone = udAllocType(udGeoZone, 1, udAF_None);

  if (m_pCurrentZone != nullptr) // In case it didn't allocate correctly
    memcpy(m_pCurrentZone, &newZone, sizeof(newZone));
}

void vcModel::ApplyDelta(vcState *pProgramState, const udDouble4x4 &delta)
{
  m_sceneMatrix = delta * m_sceneMatrix;

  // Save it to the node...
  if (m_pCurrentZone != nullptr)
  {
    udDouble3 position;
    udDouble3 scale;
    udDoubleQuat orientation;

    (udDouble4x4::translation(-m_pivot) * m_sceneMatrix * udDouble4x4::translation(m_pivot)).extractTransforms(position, scale, orientation);

    udDouble3 eulerRotation = UD_RAD2DEG(orientation.eulerAngles());

    vcProject_UpdateNodeGeometryFromCartesian(pProgramState->activeProject.pProject, m_pNode, *m_pCurrentZone, vdkPGT_Point, &position, 1);
    vdkProjectNode_SetMetadataDouble(m_pNode, "transform.rotation.y", eulerRotation.x);
    vdkProjectNode_SetMetadataDouble(m_pNode, "transform.rotation.p", eulerRotation.y);
    vdkProjectNode_SetMetadataDouble(m_pNode, "transform.rotation.r", eulerRotation.z);
    vdkProjectNode_SetMetadataDouble(m_pNode, "transform.scale", scale.x);
  }
}

void vcModel::HandleImGui(vcState * /*pProgramState*/, size_t * /*pItemID*/)
{
  ImGui::TextWrapped("Path: %s", m_pNode->pURI);
  udDouble3 position;
  udDouble3 scale;
  udDoubleQuat orientation;
  (udDouble4x4::translation(-m_pivot) * m_sceneMatrix * udDouble4x4::translation(m_pivot)).extractTransforms(position, scale, orientation);

  bool repackMatrix = false;
  if (ImGui::InputScalarN(vcString::Get("sceneModelPosition"), ImGuiDataType_Double, &position.x, 3))
    repackMatrix = true;

  udDouble3 eulerRotation = UD_RAD2DEG(orientation.eulerAngles());
  if (ImGui::InputScalarN(vcString::Get("sceneModelRotation"), ImGuiDataType_Double, &eulerRotation.x, 3))
  {
    repackMatrix = true;
    orientation = udDoubleQuat::create(UD_DEG2RAD(eulerRotation));
  }

  if (ImGui::InputScalarN(vcString::Get("sceneModelScale"), ImGuiDataType_Double, &scale.x, 1))
    repackMatrix = true;

  if (repackMatrix)
    m_sceneMatrix = udDouble4x4::translation(m_pivot) * udDouble4x4::rotationQuat(orientation, position) * udDouble4x4::scaleUniform(scale.x) * udDouble4x4::translation(-m_pivot);

  vcImGuiValueTreeObject(&m_metadata);
}

void vcModel::Cleanup(vcState *pProgramState)
{
  vdkPointCloud_Unload(pProgramState->pVDKContext, &m_pPointCloud);
  udFree(m_pCurrentZone);

  if (m_pWatermark != nullptr)
  {
    if (pProgramState->pSceneWatermark == m_pWatermark)
      pProgramState->pSceneWatermark = nullptr;

    vcTexture_Destroy(&m_pWatermark);
  }
}

udDouble3 vcModel::GetLocalSpacePivot()
{
  return m_pivot;
}

udDouble4x4 vcModel::GetWorldSpaceMatrix()
{
  return m_sceneMatrix;
}

vcGizmoAllowedControls vcModel::GetAllowedControls()
{
  return vcGAC_AllUniform;
}
