#include "vaultContext.h"
#include "vaultUDRenderer.h"
#include "vaultUDRenderView.h"
#include "vaultUDModel.h"
#include "vaultMaths.h"
#include "imgui.h"
#include "imgui_impl_sdl_gl3.h"
#include "imgui_dock.h"
#include "SDL2/SDL.h"
#include "GL/glew.h"

#include <stdlib.h>

struct Float2
{
  float x, y;
};

#define GL_VERSION_HEADER "#version 330 core\n"
#define GL_VERTEX_IN "in"
#define GL_VERTEX_OUT "out"
#define GL_FRAGMENT_IN "in"

const GLchar* const g_udFragmentShader =
"#version 330 core\n"
"precision highp float;                        \n"
"uniform sampler2D udTexture;                  \n"
"in vec2 texCoord;                \n"
"out vec4 fColor;                              \n"
"void main ()                                  \n"
"{                                             \n"
"  vec2 uv = texCoord.xy;                      \n"
"  vec4 colour = texture2D(udTexture, uv);     \n"
"  fColor = vec4(colour.bgr, 1.0);             \n"
"}                                             \n"
"";

const GLchar* const g_udVertexShader =
"#version 330 core\n"
"in vec2 vertex;                    \n"
"out vec2 texCoord;                 \n"
"void main(void)                               \n"
"{                                             \n"
"  gl_Position = vec4(vertex, 0.0, 1.0);       \n"
"  texCoord = vertex*vec2(0.5)+vec2(0.5);      \n"
"  texCoord.y = 1-texCoord.y;                  \n"
"}                                             \n"
"";

GLuint GenerateFbVBO(const Float2 *pFloats, size_t len)
{
  GLuint vboID = 0;
  // Create a new VBO and use the variable id to store the VBO id
  glGenBuffers(1, &vboID);

  // Make the new VBO active
  glBindBuffer(GL_ARRAY_BUFFER, vboID);

  // Upload vertex data to the video device
  glBufferData(GL_ARRAY_BUFFER, sizeof(Float2) * len, pFloats, GL_STATIC_DRAW); // GL_DYNAMIC_DRAW

  // Draw Triangle from VBO - do each time window, view point or data changes
  // Establish its 3 coordinates per vertex with zero stride in this array; necessary here
  //glVertexPointer(2, GL_FLOAT, 0, NULL);

  glBindBuffer(GL_ARRAY_BUFFER, 0);

  return vboID;
}

GLint glBuildShader(GLenum type, const GLchar *shaderCode)
{
  GLint compiled;
  GLint shaderCodeLen = (GLint)strlen(shaderCode);
  GLint shaderObject = glCreateShader(type);
  glShaderSource(shaderObject, 1, &shaderCode, &shaderCodeLen);
  glCompileShader(shaderObject);
  glGetShaderiv(shaderObject, GL_COMPILE_STATUS, &compiled);
  if (!compiled)
  {
    GLint blen = 1024;
    GLsizei slen = 0;
    glGetShaderiv(shaderObject, GL_INFO_LOG_LENGTH, &blen);
    if (blen > 1)
    {
      GLchar* compiler_log = (GLchar*)malloc(blen);
      glGetShaderInfoLog(shaderObject, blen, &slen, compiler_log);
      free(compiler_log);
    }
    return -1;
  }
  return shaderObject;
}

GLint glBuildProgram(GLint vertexShader, GLint fragmentShader)
{
  GLint programObject = glCreateProgram();
  if (vertexShader != -1)
    glAttachShader(programObject, vertexShader);
  if (fragmentShader != -1)
    glAttachShader(programObject, fragmentShader);

  glLinkProgram(programObject);
  GLint linked;
  glGetProgramiv(programObject, GL_LINK_STATUS, &linked);
  if (!linked)
  {
    GLint blen = 1024;
    GLsizei slen = 0;
    glGetProgramiv(programObject, GL_INFO_LOG_LENGTH, &blen);
    if (blen > 1)
    {
      GLchar* linker_log = (GLchar*)malloc(blen);
      glGetProgramInfoLog(programObject, blen, &slen, linker_log);
      free(linker_log);
    }
    return -1;
  }

  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  return programObject;
}

struct vaultContainer
{
  vaultContext *pContext;
  vaultUDRenderer *pRenderer;
  vaultUDRenderView *pRenderView;
  vaultUDModel *pModel;
};

struct RenderingState
{
  SDL_Window *pWindow;

  uint32_t *pColorBuffer;
  float *pDepthBuffer;
  GLuint texId;
  GLuint udVAO;
  GLint udProgramObject;

  GLuint framebuffer;
  GLuint texture;
  GLuint depthbuffer;

  bool planeMode;
  double deltaTime;
  vaultInt2 deltaMousePos;
  vaultMatrix camMatrix;
  vaultUint322 sceneResolution;

  bool hasContext;

  char *pServerURL;
  char *pUsername;
  char *pPassword;
        
  char *pModelPath;
};

bool vcRender(RenderingState *pRenderingState, vaultContainer *pVaultContainer);

#undef main
#ifdef SDL_MAIN_NEEDED
int SDL_main(int /*argc*/, char ** /*args*/)
#else
int main(int /*argc*/, char ** /*args*/)
#endif
{
  SDL_GLContext glcontext = NULL;
  GLint udTextureLocation = -1;
  
  const Float2 fboDataArr[] = { { -1.f,-1.f },{ -1.f,1.f },{ 1,1 },{ -1.f,-1.f },{ 1.f,-1.f },{ 1,1 } };
  GLuint fbVboId = (GLuint)-1;

  RenderingState renderingState = {};
  vaultContainer vContainer = {};

  // default values
  renderingState.planeMode = true;
  renderingState.sceneResolution.x = 1280;
  renderingState.sceneResolution.y = 720;
  renderingState.camMatrix = vaultMatrix_Identity();

  renderingState.texId = GL_INVALID_INDEX;
  renderingState.udVAO = GL_INVALID_INDEX;
  renderingState.udProgramObject = GL_INVALID_INDEX;

  // default string values.
  const char *plocalHost = "http://vau-ubu-pro-001.euclideon.local";
  const char *pUsername = "";
  const char *pPassword = "";
  const char *pModelPath = "R:\\ConvertedModels\\Aerometrex\\Aerometrix_GoldCoast_Model_1CM.uds";

  renderingState.pServerURL = new char[1024];
  renderingState.pUsername = new char[1024];
  renderingState.pPassword = new char[1024];
  renderingState.pModelPath = new char[1024];

  memset(renderingState.pServerURL, 0, 1024);
  memset(renderingState.pUsername, 0, 1024);
  memset(renderingState.pPassword, 0, 1024);
  memset(renderingState.pModelPath, 0, 1024);

  memcpy(renderingState.pServerURL, plocalHost, strlen(plocalHost));
  memcpy(renderingState.pUsername, pUsername, strlen(pUsername));
  memcpy(renderingState.pPassword, pPassword, strlen(pPassword));
  memcpy(renderingState.pModelPath, pModelPath, strlen(pModelPath));

  bool done = false;
  Uint64 NOW;
  Uint64 LAST;

  // Setup SDL
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0)
    goto epilogue;

  // Setup window
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
  if (SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3) != 0)
    goto epilogue;
  if (SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3) != 0)
    goto epilogue;

  renderingState.pWindow = SDL_CreateWindow("Euclideon Client", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
  if (!renderingState.pWindow)
    goto epilogue;

  glcontext = SDL_GL_CreateContext(renderingState.pWindow);
  if (!glcontext)
    goto epilogue;

  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK)
    goto epilogue;

  if (!ImGui_ImplSdlGL3_Init(renderingState.pWindow))
    goto epilogue;

  renderingState.udProgramObject = glBuildProgram(glBuildShader(GL_VERTEX_SHADER, g_udVertexShader), glBuildShader(GL_FRAGMENT_SHADER, g_udFragmentShader));
  if (renderingState.udProgramObject == -1)
    goto epilogue;

  glUseProgram(renderingState.udProgramObject);

  udTextureLocation = glGetUniformLocation(renderingState.udProgramObject, "udTexture");
  glUniform1i(udTextureLocation, 0);

  glGenVertexArrays(1, &renderingState.udVAO);
  glBindVertexArray(renderingState.udVAO);

  fbVboId = GenerateFbVBO(fboDataArr, 6);

  glBindBuffer(GL_ARRAY_BUFFER, fbVboId);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, false, sizeof(float) * 2, 0);

  glBindVertexArray(0);
  glUseProgram(0);

  glGenTextures(1, &renderingState.texId);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, renderingState.texId);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);

  renderingState.pColorBuffer = new uint32_t[renderingState.sceneResolution.x*renderingState.sceneResolution.y];
  renderingState.pDepthBuffer = new float[renderingState.sceneResolution.x*renderingState.sceneResolution.y];

  glGenFramebuffers(1, &renderingState.framebuffer);
  glBindFramebuffer(GL_FRAMEBUFFER, renderingState.framebuffer);

  glGenRenderbuffers(1, &renderingState.depthbuffer);
  glBindRenderbuffer(GL_RENDERBUFFER, renderingState.depthbuffer);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, renderingState.sceneResolution.x, renderingState.sceneResolution.y);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, renderingState.depthbuffer);

  glGenTextures(1, &renderingState.texture);
  glBindTexture(GL_TEXTURE_2D, renderingState.texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, renderingState.sceneResolution.x, renderingState.sceneResolution.y, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderingState.texture, 0);
  GLenum DrawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
  glDrawBuffers(1, DrawBuffers);

  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
  {
    //TODO: Handle this properly
  }

  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  //Get ready...
  NOW = SDL_GetPerformanceCounter();
  LAST = 0;

  ImGui::LoadDock();

  while (!done)
  {
    SDL_Event event;
    while (SDL_PollEvent(&event))
    {
      ImGui_ImplSdlGL3_ProcessEvent(&event);
      done = event.type == SDL_QUIT;
    }

    LAST = NOW;
    NOW = SDL_GetPerformanceCounter();
    renderingState.deltaTime = double(NOW - LAST) / SDL_GetPerformanceFrequency();

    SDL_GetRelativeMouseState(&renderingState.deltaMousePos.x, &renderingState.deltaMousePos.y);

    ImGui_ImplSdlGL3_NewFrame(renderingState.pWindow);

    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, renderingState.sceneResolution.x, renderingState.sceneResolution.y);

    if (!vcRender(&renderingState, &vContainer))
      goto epilogue;

    ImGui::Render();
    SDL_GL_SwapWindow(renderingState.pWindow);
  }

  ImGui::SaveDock();

epilogue:
  vaultUDModel_Unload(vContainer.pContext, &vContainer.pModel);

  vaultUDRenderView_Destroy(vContainer.pContext, &vContainer.pRenderView);
  vaultUDRenderer_Destroy(vContainer.pContext, &vContainer.pRenderer);

  vaultContext_Disconnect(&vContainer.pContext);
  delete[] renderingState.pColorBuffer;
  delete[] renderingState.pDepthBuffer;

  delete[] renderingState.pServerURL;
  delete[] renderingState.pUsername;
  delete[] renderingState.pPassword;
  delete[] renderingState.pModelPath;

  return 0;
}

void vcRenderScene(RenderingState *pRenderingState, vaultContainer *pVaultContainer)
{
  //Rendering
  ImVec2 size = ImGui::GetContentRegionAvail();
  ImGuiIO& io = ImGui::GetIO();
  ImVec2 cursorPos = ImGui::GetCursorScreenPos();          // "cursor" is where imgui will draw the image
  ImVec2 mousePos = io.MousePos;

  if (pRenderingState->sceneResolution.x != size.x || pRenderingState->sceneResolution.y != size.y) //Resize buffers
  {
    pRenderingState->sceneResolution.x = (uint32_t)size.x;
    pRenderingState->sceneResolution.y = (uint32_t)size.y;

    //Resize GPU Targets
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, pRenderingState->texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, pRenderingState->sceneResolution.x, pRenderingState->sceneResolution.y, 0, GL_RED, GL_UNSIGNED_BYTE, NULL);
    glBindFramebuffer(GL_FRAMEBUFFER, pRenderingState->framebuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, pRenderingState->sceneResolution.x, pRenderingState->sceneResolution.y);

    //Resize CPU Targets
    vaultUDRenderView_Destroy(pVaultContainer->pContext, &pVaultContainer->pRenderView);

    delete pRenderingState->pColorBuffer;
    delete pRenderingState->pDepthBuffer;

    pRenderingState->pColorBuffer = new uint32_t[pRenderingState->sceneResolution.x*pRenderingState->sceneResolution.y];
    pRenderingState->pDepthBuffer = new float[pRenderingState->sceneResolution.x*pRenderingState->sceneResolution.y];

    //TODO: Error Detection
    vaultUDRenderView_Create(pVaultContainer->pContext, &pVaultContainer->pRenderView, pVaultContainer->pRenderer, pRenderingState->sceneResolution.x, pRenderingState->sceneResolution.y);
    vaultUDRenderView_SetTargets(pVaultContainer->pContext, pVaultContainer->pRenderView, pRenderingState->pColorBuffer, 0, pRenderingState->pDepthBuffer);
    vaultUDRenderView_SetMatrix(pVaultContainer->pContext, pVaultContainer->pRenderView, vUDRVM_Camera, pRenderingState->camMatrix.a);
  }

  bool isHovered = ImGui::IsItemHovered();
  bool isLeftClicked = ImGui::IsMouseClicked(0, false);
  bool isRightClicked = ImGui::IsMouseClicked(1, false);
  bool isFocused = ImGui::IsWindowFocused();

  const Uint8 *pKeysArray = SDL_GetKeyboardState(NULL);

  int forwardMovement = (int)pKeysArray[SDL_SCANCODE_W] - (int)pKeysArray[SDL_SCANCODE_S];
  int rightMovement = (int)pKeysArray[SDL_SCANCODE_D] - (int)pKeysArray[SDL_SCANCODE_A];
  int upMovement = (int)pKeysArray[SDL_SCANCODE_R] - (int)pKeysArray[SDL_SCANCODE_F];

  vaultDouble4 direction;
  if (pRenderingState->planeMode)
  {
    direction = pRenderingState->camMatrix.axis.y * forwardMovement + pRenderingState->camMatrix.axis.x * rightMovement + vaultDouble4{ 0, 0, (double)upMovement, 0 }; // don't use the camera orientation
  }
  else
  {
    direction = pRenderingState->camMatrix.axis.y * forwardMovement + pRenderingState->camMatrix.axis.x * rightMovement;
    direction.z = 0;
    direction += vaultDouble4{ 0, 0, (double)upMovement, 0 }; // don't use the camera orientation
  }

  pRenderingState->camMatrix.axis.t += direction * pRenderingState->deltaTime;

  if (SDL_GetMouseState(NULL, NULL) & SDL_BUTTON(SDL_BUTTON_LEFT) && !ImGui::IsMouseHoveringAnyWindow())
  {
    vaultDouble4 translation = pRenderingState->camMatrix.axis.t;
    pRenderingState->camMatrix.axis.t = { 0,0,0,1 };
    pRenderingState->camMatrix = vaultMatrix_RotationAxis({ 0,0,-1 }, pRenderingState->deltaMousePos.x / 100.0) * pRenderingState->camMatrix; // rotate on global axis and add back in the position
    pRenderingState->camMatrix.axis.t = translation;
    pRenderingState->camMatrix *= vaultMatrix_RotationAxis({ -1,0,0 }, pRenderingState->deltaMousePos.y / 100.0); // rotate on local axis, since this is the only one there will be no problem
  }

  vaultError err = vaultUDRenderView_SetMatrix(pVaultContainer->pContext, pVaultContainer->pRenderView, vUDRVM_Camera, pRenderingState->camMatrix.a);
  if (err != vE_Success)
    goto epilogue;

  if (pVaultContainer->pModel != nullptr)
  {
    err = vaultUDRenderer_Render(pVaultContainer->pContext, pVaultContainer->pRenderer, pVaultContainer->pRenderView, &pVaultContainer->pModel, 1);
    if (err != vE_Success)
      goto epilogue;
  }

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, pRenderingState->texId);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, pRenderingState->sceneResolution.x, pRenderingState->sceneResolution.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, pRenderingState->pColorBuffer);
  
  glBindFramebuffer(GL_FRAMEBUFFER, pRenderingState->framebuffer);
  glViewport(0, 0, pRenderingState->sceneResolution.x, pRenderingState->sceneResolution.y);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glUseProgram(pRenderingState->udProgramObject);
  glBindTexture(GL_TEXTURE_2D, pRenderingState->texId);
  glBindVertexArray(pRenderingState->udVAO);

  glDrawArrays(GL_TRIANGLES, 0, 6);

  glBindVertexArray(0);
  glUseProgram(0);
  glBindTexture(GL_TEXTURE_2D, 0);

  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  ImGui::Image((ImTextureID)((size_t)pRenderingState->texture), size, ImVec2(0, 0), ImVec2(1, -1));

epilogue:
  return;
}

int vcMainMenuGui()
{
  int menuHeight = 0;

  if (ImGui::BeginMainMenuBar())
  {
    if (ImGui::BeginMenu("File"))
    {
      if (ImGui::MenuItem("New"))
      {

      }

      if (ImGui::MenuItem("Open", "Ctrl+O"))
      {

      }

      if (ImGui::BeginMenu("Open Recent"))
      {
        //ImGui::MenuItem("fish_hat.h");
        ImGui::EndMenu();
      }

      if (ImGui::MenuItem("Quit", "Alt+F4"))
      {

      }

      ImGui::EndMenu();
    }

    menuHeight = (int)ImGui::GetWindowSize().y;

    ImGui::EndMainMenuBar();
  }

  return menuHeight;
}

bool vcRender(RenderingState *pRenderingState, vaultContainer *pVaultContainer)
{
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  vaultError err;

  bool result = false;

  int menuHeight = (!pRenderingState->hasContext) ? 0 : vcMainMenuGui();

  if (ImGui::GetIO().DisplaySize.y > 0)
  {
    ImVec2 pos = ImVec2(0.f, (float)menuHeight);
    ImVec2 size = ImGui::GetIO().DisplaySize;
    size.y -= pos.y;
    ImGui::RootDock(pos, ImVec2(size.x, size.y - 25.0f));

    // Draw status bar (no docking)
    ImGui::SetNextWindowSize(ImVec2(size.x, 25.0f), ImGuiSetCond_Always);
    ImGui::SetNextWindowPos(ImVec2(0, size.y - 6.0f), ImGuiSetCond_Always);
    ImGui::Begin("statusbar", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoResize);
    ImGui::Text("FPS: %f", ImGui::GetIO().Framerate);
    ImGui::End();
  }

  if (!pRenderingState->hasContext)
  {
    if (ImGui::Begin("Login"))
    {
      ImGui::InputText("ServerURL", pRenderingState->pServerURL, 1024);
      ImGui::InputText("Username", pRenderingState->pUsername, 1024);
      ImGui::InputText("Password", pRenderingState->pPassword, 1024);

      if (ImGui::Button("Login!"))
      {
        err = vaultContext_Connect(&pVaultContainer->pContext, pRenderingState->pServerURL, "ClientSample");
        if (err != vE_Success)
          goto epilogue;

        err = vaultContext_Login(pVaultContainer->pContext, pRenderingState->pUsername, pRenderingState->pPassword);
        if (err != vE_Success)
          goto epilogue;

        err = vaultContext_GetLicense(pVaultContainer->pContext, vaultLT_Basic);
        if (err != vE_Success)
          goto epilogue;

        err = vaultUDRenderer_Create(pVaultContainer->pContext, &pVaultContainer->pRenderer);
        if (err != vE_Success)
          goto epilogue;

        err = vaultUDRenderView_Create(pVaultContainer->pContext, &pVaultContainer->pRenderView, pVaultContainer->pRenderer, pRenderingState->sceneResolution.x, pRenderingState->sceneResolution.y);
        if (err != vE_Success)
          goto epilogue;

        pRenderingState->hasContext = true;
      }

      if (ImGui::Button("Guest Login!"))
      {
        err = vaultContext_Connect(&pVaultContainer->pContext, pRenderingState->pServerURL, "ClientSample");
        if (err != vE_Success)
          goto epilogue;

        err = vaultContext_GetLicense(pVaultContainer->pContext, vaultLT_Basic);
        if (err != vE_Success)
          goto epilogue;

        err = vaultUDRenderer_Create(pVaultContainer->pContext, &pVaultContainer->pRenderer);
        if (err != vE_Success)
          goto epilogue;

        err = vaultUDRenderView_Create(pVaultContainer->pContext, &pVaultContainer->pRenderView, pVaultContainer->pRenderer, pRenderingState->sceneResolution.x, pRenderingState->sceneResolution.y);
        if (err != vE_Success)
          goto epilogue;

        pRenderingState->hasContext = true;
      }
    }

    ImGui::End();
  }
  else
  {
    if (ImGui::BeginDock("Scene", nullptr, ImGuiWindowFlags_NoScrollbar))
      vcRenderScene(pRenderingState, pVaultContainer);
    ImGui::EndDock();

    if (ImGui::BeginDock("Dummy3"))
      ImGui::Text("Placeholder!");
    ImGui::EndDock();

    if (ImGui::BeginDock("StyleEditor"))
      ImGui::ShowStyleEditor();
    ImGui::EndDock();

    //Load Model
    if (ImGui::BeginDock("Load Model"))
    {
      ImGui::InputText("Model Path", pRenderingState->pModelPath, 1024);

      if (ImGui::Button("Load Model!"))
      {
        err = vaultUDModel_Load(pVaultContainer->pContext, &pVaultContainer->pModel, pRenderingState->pModelPath);
        if (err != vE_Success)
          goto epilogue;
      }
    }
    ImGui::EndDock();

    if (ImGui::BeginDock("Settings"))
    {
      if (ImGui::Button("Unload Model"))
      {
        err = vaultUDModel_Unload(pVaultContainer->pContext, &pVaultContainer->pModel);
        if (err != vE_Success)
          goto epilogue;
      }

      if (ImGui::Button("Logout"))
      {
        err = vaultUDModel_Unload(pVaultContainer->pContext, &pVaultContainer->pModel);
        if (err != vE_Success)
          goto epilogue;

        err = vaultContext_Logout(pVaultContainer->pContext);
        if (err != vE_Success)
          goto epilogue;

        err = vaultContext_GetLicense(pVaultContainer->pContext, vaultLT_Basic);
        if (err != vE_Success)
          goto epilogue;

        pRenderingState->hasContext = false;
      }

      if (ImGui::RadioButton("PlaneMode", pRenderingState->planeMode))
        pRenderingState->planeMode = true;
      if (ImGui::RadioButton("HeliMode", !pRenderingState->planeMode))
        pRenderingState->planeMode = false;
    }
    ImGui::EndDock();

  }

  result = true;

epilogue:
  return result;
}
