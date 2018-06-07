
#include "vcTexture.h"
#include "vcRenderUtils.h"
#include "vcSettings.h"
#include "udPlatform/udFile.h"
#include "udPlatform/udPlatformUtil.h"
#include "stb_image.h"

bool vcTexture_Create(vcTexture **ppTexture, uint32_t width, uint32_t height, vcTextureFormat format /*= vcTextureFormat_RGBA8*/, GLuint filterMode /*= GL_NEAREST*/, bool hasMipmaps /*= false*/, uint8_t *pPixels /*= nullptr*/, int32_t aniFilter /*= 0*/, int32_t wrapMode /*= GL_REPEAT*/)
{
  if (ppTexture == nullptr || width == 0 || height == 0)
    return false;

  vcTexture *pTexture = udAllocType(vcTexture, 1, udAF_Zero);

  glGenTextures(1, &pTexture->id);
  glBindTexture(GL_TEXTURE_2D, pTexture->id);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filterMode);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, hasMipmaps ? GL_LINEAR_MIPMAP_LINEAR : filterMode);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapMode);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapMode);

  if (aniFilter > 0)
  {
    float aniso;
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &aniso);
    aniso = udMin((float)aniFilter, aniso);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, aniso);
  }

  GLint internalFormat;
  switch (format)
  {
  case vcTextureFormat_RGBA8: // fall through
  default:
    internalFormat = GL_RGBA8;
  }

  glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, pPixels);

  if (hasMipmaps)
    glGenerateMipmap(GL_TEXTURE_2D);

  glBindTexture(GL_TEXTURE_2D, 0);
  VERIFY_GL();

  pTexture->format = format;
  pTexture->width = width;
  pTexture->height = height;

  *ppTexture = pTexture;
  return true;
}

bool vcTexture_CreateDepth(vcTexture **ppTexture, uint32_t width, uint32_t height, vcTextureFormat format /*= vcTextureFormat_D24*/, GLuint filterMode /*= GL_NEAREST*/)
{
  if (ppTexture == nullptr || width == 0 || height == 0)
    return false;

  vcTexture *pTexture = udAllocType(vcTexture, 1, udAF_Zero);

  glGenTextures(1, &pTexture->id);
  glBindTexture(GL_TEXTURE_2D, pTexture->id);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filterMode);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filterMode);
  VERIFY_GL();

  GLint internalFormat;
  GLenum type;
  switch (format)
  {
  case vcTextureFormat_D32F:
    internalFormat = GL_DEPTH_COMPONENT32F;
    type = GL_FLOAT;
    break;
  case vcTextureFormat_D24: // fall through
  default:
    internalFormat = GL_DEPTH_COMPONENT24;
    type = GL_UNSIGNED_INT;
  }

  glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, GL_DEPTH_COMPONENT, type, NULL);

  glBindTexture(GL_TEXTURE_2D, 0);
  VERIFY_GL();

  pTexture->format = format;
  pTexture->width = width;
  pTexture->height = height;

  *ppTexture = pTexture;

  return true;
}

bool vcTexture_CreateFromFilename(vcTexture **ppTexture, const char *pFilename, uint32_t *pWidth /*= nullptr*/, uint32_t *pHeight /*= nullptr*/, int32_t filterMode /*= GL_LINEAR*/, bool hasMipmaps /*= false*/, int32_t aniFilter /*= 0*/, int32_t wrapMode /*= GL_REPEAT*/)
{
  if (ppTexture == nullptr || pFilename == nullptr)
    return false;

  uint32_t width, height, channelCount;
  vcTexture *pTexture = nullptr;

  void *pFileData;
  int64_t fileLen;

  if (udFile_Load(pFilename, &pFileData, &fileLen) != udR_Success)
    return false;

  uint8_t *pData = stbi_load_from_memory((stbi_uc*)pFileData, (int)fileLen, (int*)&width, (int*)&height, (int*)&channelCount, 4);
  udFree(pFileData);

  if (pData)
    vcTexture_Create(&pTexture, width, height, vcTextureFormat_RGBA8, filterMode, hasMipmaps, pData, aniFilter, wrapMode);

  stbi_image_free(pData);

  if (pWidth != nullptr)
    *pWidth = width;

  if (pHeight != nullptr)
    *pHeight = height;

  *ppTexture = pTexture;

  return (pTexture != nullptr);
}

void vcTexture_UploadPixels(vcTexture *pTexture, const void *pPixels, int width, int height)
{
  GLenum internalFormat;
  GLenum pixelFormat, pixelType;
  switch (pTexture->format)
  {
    case vcTextureFormat_RGBA8:
      internalFormat = GL_RGBA;
      pixelFormat = GL_RGBA;
      pixelType = GL_UNSIGNED_BYTE;
      break;
    case vcTextureFormat_D24:
      internalFormat = GL_DEPTH_COMPONENT24;
      pixelFormat = GL_DEPTH_COMPONENT;
      pixelType = GL_FLOAT;
      break;
    case vcTextureFormat_D32F:
      internalFormat = GL_DEPTH_COMPONENT32F;
      pixelFormat = GL_DEPTH_COMPONENT;
      pixelType = GL_FLOAT;
      break;
    default:
      // unknown texture format, do nothing
      return;
  }

  pTexture->width = width;
  pTexture->height = height;

  glBindTexture(GL_TEXTURE_2D, pTexture->id);
  glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, pTexture->width, pTexture->height, 0, pixelFormat, pixelType, pPixels);
  glBindTexture(GL_TEXTURE_2D, 0);
}

void vcTexture_Destroy(vcTexture **ppTexture)
{
  if (ppTexture == nullptr || *ppTexture == nullptr)
    return;

  glDeleteTextures(1, &(*ppTexture)->id);
  udFree(*ppTexture);
}

bool vcTexture_LoadCubemap(vcTexture **ppTexture, const char *pFilename)
{
  vcTexture *pTexture = udAllocType(vcTexture, 1, udAF_Zero);
  udFilename fileName(pFilename);

  pTexture->format = vcTextureFormat_Cubemap;

  glGenTextures(1, &pTexture->id);
  glBindTexture(GL_TEXTURE_CUBE_MAP, pTexture->id);

  const char* names[] = { "_LF", "_RT", "_FR", "_BK", "_UP", "_DN" };
  const GLenum types[] = { GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_X, GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, GL_TEXTURE_CUBE_MAP_POSITIVE_Z, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z };

  size_t filenameLen = udStrlen(pFilename);
#if UDPLATFORM_IOS || UDPLATFORM_IOS_SIMULATOR
  const char skyboxPath[] = ASSETDIR;
#elif UDPLATFORM_OSX

  char *pSkyboxPath;
  char *pBaseDir = SDL_GetBasePath();
  if (pBaseDir)
    pSkyboxPath = pBaseDir;
  else
    pSkyboxPath = SDL_strdup("./");
  char skyboxPath[1024] = "";
  udSprintf(skyboxPath, 1024, "%s", pSkyboxPath);
#else
  const char skyboxPath[] = ASSETDIR "skyboxes/";
#endif
  size_t pathLen = udStrlen(skyboxPath);
  char *pFilePath = udStrdup(skyboxPath, filenameLen + 5);

  for (int i = 0; i < 6; i++) // for each face of the cube map
  {
    int width, height, depth;

    char fileNameNoExt[256] = "";
    fileName.ExtractFilenameOnly(fileNameNoExt,UDARRAYSIZE(fileNameNoExt));
    udSprintf(pFilePath, filenameLen + 5 + pathLen, "%s%s%s%s", skyboxPath, fileNameNoExt, names[i], fileName.GetExt());
    uint8_t* data = stbi_load(pFilePath, &width, &height, &depth, 0);

    pTexture->height = height;
    pTexture->width = width;

    if (data)
    {
      if (depth == 3)
        glTexImage2D(types[i], 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
      else
        glTexImage2D(types[i], 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);

      stbi_image_free(data);
    }
    VERIFY_GL();
  }

  udFree(pFilePath);

  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

  UDASSERT(tex.id != GL_INVALID_INDEX, "Didn't load cubemap correctly!");
  VERIFY_GL();

#if UDPLATFORM_OSX
  SDL_free(pBaseDir);
#endif

  *ppTexture = pTexture;
  return true;
}
