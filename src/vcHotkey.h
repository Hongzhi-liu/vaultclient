#ifndef vcHotkey_h__
#define vcHotkey_h__

#include "udResult.h"
#include "vcState.h"

enum modifierFlags
{
  vcMOD_Shift = 1024,
  vcMOD_Ctrl = 2048,
  vcMOD_Alt = 4096,
  vcMOD_Super = 8192
};

enum vcBind
{
  // Down bindings
  vcB_Forward,
  vcB_Backward,
  vcB_Left,
  vcB_Up,
  vcB_Right,
  vcB_Down,

  // Other Bindings
  vcB_Remove,
  vcB_Close,
  vcB_LockAltitude,
  vcB_GizmoTranslate,
  vcB_GizmoRotate,
  vcB_GizmoScale,
  vcB_GizmoLocalSpace,
  vcB_Fullscreen,
  vcB_RenameSceneItem,
  vcB_Save,
  vcB_Load,
  vcB_AddUDS,
  vcB_BindingsInterface,
  vcB_Count
};

namespace vcHotkey
{
  bool IsDown(int keyNum);
  bool IsDown(vcBind key);
  bool IsPressed(int keyNum, bool unique = true);
  bool IsPressed(vcBind key, bool unique = true);
  void GetKeyName(vcBind key, char *pBuffer, uint32_t bufferLen);

  template <size_t N>
  void GetKeyName(vcBind key, char(&buffer)[N])
  {
    GetKeyName(key, buffer, N);
  };

  const char* GetBindName(vcBind key);
  vcBind BindFromName(const char* pName);
  int GetMod(int key);
  void Set(vcBind key, int value);
  int Get(vcBind key);

  void DisplayBindings(vcState *pProgramState);
  int DecodeKeyString(const char* pBind);
}

#endif //vcHotkey_h__
