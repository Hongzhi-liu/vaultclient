#include "vcImGuiSimpleWidgets.h"

#include "udPlatform.h"
#include "udPlatformUtil.h"
#include "udStringUtil.h"

#include "imgui.h"
#include "imgui_internal.h"

#include "vcStrings.h"

struct vcIGSWResizeContainer
{
  char **ppBuffer;
  size_t *pBufferSize;
};

int vcIGSW_ResizeString(ImGuiInputTextCallbackData *pData)
{
  if (pData->EventFlag == ImGuiInputTextFlags_CallbackResize)
  {
    vcIGSWResizeContainer *pInfo = (vcIGSWResizeContainer*)pData->UserData;

    size_t expectedBufSize = (size_t)pData->BufSize;
    size_t currentBufSize = *pInfo->pBufferSize;

    if (expectedBufSize <= currentBufSize)
      return 0; // We don't need to resize right now

    size_t additionalBytes = expectedBufSize - currentBufSize;
    additionalBytes = udMin(additionalBytes * 2, additionalBytes + 64); // Gives double the amount requested until the amount requested is more than 32

    char *pNewStr = (char*)udMemDup(*pInfo->ppBuffer, currentBufSize, additionalBytes, udAF_Zero);
    udFree(*pInfo->ppBuffer);
    *pInfo->ppBuffer = pNewStr;
    *pInfo->pBufferSize = currentBufSize + additionalBytes;

    pData->Buf = pNewStr;
  }

  return 0;
}

bool vcIGSW_InputTextWithResize(const char *pLabel, char **ppBuffer, size_t *pBufferSize, ImGuiInputTextFlags flags /*= ImGuiInputTextFlags_None*/)
{
  vcIGSWResizeContainer info;
  info.ppBuffer = ppBuffer;
  info.pBufferSize = pBufferSize;

  if (*pBufferSize == 0)
    *pBufferSize = udStrlen(*ppBuffer) + 1; //+1 for '\0'

  return ImGui::InputText(pLabel, *ppBuffer, *pBufferSize, ImGuiInputTextFlags_CallbackResize | flags, vcIGSW_ResizeString, &info);
}

bool vcIGSW_ColorPickerU32(const char *pLabel, uint32_t *pColor, ImGuiColorEditFlags flags)
{
  float colors[4];

  colors[0] = ((((*pColor) >> 16) & 0xFF) / 255.f); // Blue
  colors[1] = ((((*pColor) >> 8) & 0xFF) / 255.f); // Green
  colors[2] = ((((*pColor) >> 0) & 0xFF) / 255.f); // Red
  colors[3] = ((((*pColor) >> 24) & 0xFF) / 255.f); // Alpha

  if (ImGui::ColorEdit4(pLabel, colors, flags))
  {
    uint32_t val = 0;

    val |= ((int)(colors[0] * 255) << 16); // Blue
    val |= ((int)(colors[1] * 255) << 8); // Green
    val |= ((int)(colors[2] * 255) << 0); // Red
    val |= ((int)(colors[3] * 255) << 24); // Alpha

    *pColor = val;

    return true;
  }

  return false;
}

udFloat4 vcIGSW_BGRAToImGui(uint32_t lineColour)
{
  //TODO: Find or add a math helper for this
  udFloat4 colours; // RGBA
  colours.x = ((((lineColour) >> 16) & 0xFF) / 255.f); // Red
  colours.y = ((((lineColour) >> 8) & 0xFF) / 255.f); // Green
  colours.z = ((((lineColour) >> 0) & 0xFF) / 255.f); // Blue
  colours.w = ((((lineColour) >> 24) & 0xFF) / 255.f); // Alpha

  return colours;
}

uint32_t vcIGSW_BGRAToRGBAUInt32(uint32_t lineColour)
{
  // BGRA to RGBA
  return ((lineColour & 0xff) << 16) | (lineColour & 0x0000ff00) | (((lineColour >> 16) & 0xff) << 0) | (lineColour & 0xff000000);
}

bool vcIGSW_IsItemHovered(ImGuiHoveredFlags /*flags = 0*/, float timer /*= 0.5f*/)
{
  return ImGui::IsItemHovered() && GImGui->HoveredIdTimer > timer;
}

void vcIGSW_ShowLoadStatusIndicator(vcSceneLoadStatus loadStatus, bool sameLine /*= true*/)
{
  const char *loadingChars[] = { "\xE2\x96\xB2", "\xE2\x96\xB6", "\xE2\x96\xBC", "\xE2\x97\x80" };
  int64_t currentLoadingChar = (int64_t)(10 * udGetEpochSecsUTCf());

  // Load Status (if any)
  if (loadStatus == vcSLS_Pending)
  {
    ImGui::TextColored(ImVec4(1.f, 1.f, 0.f, 1.f), "\xE2\x9A\xA0"); // Yellow Exclamation in Triangle
    if (vcIGSW_IsItemHovered())
      ImGui::SetTooltip("%s", vcString::Get("sceneExplorerPending"));

    if (sameLine)
      ImGui::SameLine();
  }
  else if (loadStatus == vcSLS_Loading)
  {
    ImGui::TextColored(ImVec4(1.f, 1.f, 0.f, 1.f), "%s", loadingChars[currentLoadingChar % udLengthOf(loadingChars)]); // Yellow Spinning clock
    if (vcIGSW_IsItemHovered())
      ImGui::SetTooltip("%s", vcString::Get("sceneExplorerLoading"));

    if (sameLine)
      ImGui::SameLine();
  }
  else if (loadStatus == vcSLS_Failed || loadStatus == vcSLS_OpenFailure)
  {
    ImGui::TextColored(ImVec4(1.f, 0.f, 0.f, 1.f), "\xE2\x9A\xA0"); // Red Exclamation in Triangle
    if (vcIGSW_IsItemHovered())
    {
      if (loadStatus == vcSLS_OpenFailure)
        ImGui::SetTooltip("%s", vcString::Get("sceneExplorerErrorOpen"));
      else
        ImGui::SetTooltip("%s", vcString::Get("sceneExplorerErrorLoad"));
    }

    if (sameLine)
      ImGui::SameLine();
  }
}

bool vcIGSW_StickyIntSlider(const char* label, int* v, int v_min, int v_max, int sticky)
{
  int stickThreshold = v_max / 500;

  if (*v > (v_max - stickThreshold))
    *v = v_max;
  else if (*v < stickThreshold)
    *v = v_min;
  else if (*v >(sticky - stickThreshold) && *v < (sticky + stickThreshold))
    *v = sticky;

  if (ImGui::SliderInt(label, v, v_min, v_max, "%d"))
  {
    *v = udClamp(*v, v_min, v_max);
    return true;
  }
  return false;
}
