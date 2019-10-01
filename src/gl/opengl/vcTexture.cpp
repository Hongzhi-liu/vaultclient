#include "vcOpenGL.h"

#include "vcSettings.h"
#include "gl/vcTexture.h"
#include "gl/vcFramebuffer.h"

#include "udFile.h"
#include "udPlatformUtil.h"
#include "udStringUtil.h"

#include "stb_image.h"

void vcTexture_GetFormatAndPixelSize(const vcTextureFormat format, int *pPixelSize = nullptr, GLint *pTextureFormat = nullptr, GLenum *pPixelType = nullptr, GLint *pPixelFormat = nullptr)
{
  GLint textureFormat = GL_INVALID_ENUM;
  GLenum pixelType = GL_INVALID_ENUM;
  GLint pixelFormat = GL_INVALID_ENUM;
  int pixelSize = 0; // in bytes

  switch (format)
  {
  case vcTextureFormat_RGBA8:
    textureFormat = GL_RGBA8;
    pixelType = GL_UNSIGNED_BYTE;
    pixelFormat = GL_RGBA;
    pixelSize = 4;
    break;
  case vcTextureFormat_BGRA8:
    textureFormat = GL_RGBA8;
    pixelType = GL_UNSIGNED_BYTE;
#if UDPLATFORM_EMSCRIPTEN
    glFormat = GL_RGBA; // TODO: Fix this
#else
    pixelFormat = GL_BGRA;
#endif
    pixelSize = 4;
    break;
  case vcTextureFormat_D32F:
    textureFormat = GL_DEPTH_COMPONENT32F;
    pixelType = GL_FLOAT;
    pixelFormat = GL_DEPTH_COMPONENT;
    pixelSize = 4;
    break;
  case vcTextureFormat_D24S8:
    textureFormat = GL_DEPTH24_STENCIL8;
    pixelType = GL_UNSIGNED_INT_24_8;
    pixelFormat = GL_DEPTH_STENCIL;
    pixelSize = 4;
    break;

  case vcTextureFormat_Unknown: // fall through
  case vcTextureFormat_Cubemap: // fall through
  case vcTextureFormat_Count:
    break;
  }

  if (pPixelSize != nullptr)
    *pPixelSize = pixelSize;

  if (pTextureFormat != nullptr)
    *pTextureFormat = textureFormat;

  if (pPixelType != nullptr)
    *pPixelType = pixelType;

  if (pPixelFormat != nullptr)
    *pPixelFormat = pixelFormat;
}

udResult vcTexture_Create(vcTexture **ppTexture, uint32_t width, uint32_t height, const void *pPixels, vcTextureFormat format /*= vcTextureFormat_RGBA8*/, vcTextureFilterMode filterMode /*= vcTFM_Nearest*/, bool hasMipmaps /*= false*/, vcTextureWrapMode wrapMode /*= vcTWM_Repeat*/, vcTextureCreationFlags flags /*= vcTCF_None*/, int32_t aniFilter /* = 0 */)
{
  if (ppTexture == nullptr || width == 0 || height == 0 || format == vcTextureFormat_Unknown || format == vcTextureFormat_Count || format == vcTextureFormat_Cubemap)
    return udR_InvalidParameter_;

  udResult result = udR_Success;
  GLint textureFormat = GL_INVALID_ENUM;
  GLenum pixelType = GL_INVALID_ENUM;
  GLint pixelFormat = GL_INVALID_ENUM;
  int pixelBytes = 0;
  vcTexture_GetFormatAndPixelSize(format, &pixelBytes, &textureFormat, &pixelType, &pixelFormat);

  vcTexture *pTexture = udAllocType(vcTexture, 1, udAF_Zero);
  UD_ERROR_NULL(pTexture, udR_MemoryAllocationFailure);

  glGenTextures(1, &pTexture->id);
  glBindTexture(GL_TEXTURE_2D, pTexture->id);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, vcTFMToGL[filterMode]);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, hasMipmaps ? GL_LINEAR_MIPMAP_NEAREST : vcTFMToGL[filterMode]);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, vcTWMToGL[wrapMode]);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, vcTWMToGL[wrapMode]);

  if (aniFilter > 0)
  {
    int32_t realAniso = vcGLState_GetMaxAnisotropy(aniFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, realAniso);
  }

  glTexImage2D(GL_TEXTURE_2D, 0, textureFormat, width, height, 0, pixelFormat, pixelType, pPixels);

  if (hasMipmaps)
    glGenerateMipmap(GL_TEXTURE_2D);

  if ((flags & vcTCF_AsynchronousRead) == vcTCF_AsynchronousRead)
  {
    glGenBuffers(2, pTexture->pbos);
    for (int i = 0; i < 2; ++i)
    {
      glBindBuffer(GL_PIXEL_PACK_BUFFER, pTexture->pbos[i]);
      glBufferData(GL_PIXEL_PACK_BUFFER, width * height * pixelBytes, 0, GL_STREAM_READ);
    }
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
  }

  glBindTexture(GL_TEXTURE_2D, 0);

  pTexture->flags = flags;
  pTexture->format = format;
  pTexture->width = width;
  pTexture->height = height;

  vcGLState_ReportGPUWork(0, 0, pTexture->width * pTexture->height * pixelBytes);

  *ppTexture = pTexture;
  pTexture = nullptr;

epilogue:
  if (pTexture != nullptr)
    vcTexture_Destroy(&pTexture);

  VERIFY_GL();
  return result;

}

bool vcTexture_CreateFromMemory(vcTexture **ppTexture, void *pFileData, size_t fileLength, uint32_t *pWidth /*= nullptr*/, uint32_t *pHeight /*= nullptr*/, vcTextureFilterMode filterMode /*= vcTFM_Linear*/, bool hasMipmaps /*= false*/, vcTextureWrapMode wrapMode /*= vcTWM_Repeat*/, vcTextureCreationFlags flags /*= vcTCF_None*/, int32_t aniFilter /*= 0*/)
{
  if (ppTexture == nullptr || pFileData == nullptr || fileLength == 0)
    return false;

  uint32_t width, height, channelCount;
  vcTexture *pTexture = nullptr;

  uint8_t *pData = stbi_load_from_memory((stbi_uc*)pFileData, (int)fileLength, (int*)&width, (int*)&height, (int*)&channelCount, 4);

  if (pData)
    vcTexture_Create(&pTexture, width, height, pData, vcTextureFormat_RGBA8, filterMode, hasMipmaps, wrapMode, flags, aniFilter);

  stbi_image_free(pData);

  if (pWidth != nullptr)
    *pWidth = width;

  if (pHeight != nullptr)
    *pHeight = height;

  *ppTexture = pTexture;

  return (pTexture != nullptr);
}

bool vcTexture_CreateFromFilename(vcTexture **ppTexture, const char *pFilename, uint32_t *pWidth /*= nullptr*/, uint32_t *pHeight /*= nullptr*/, vcTextureFilterMode filterMode /*= vcTFM_Linear*/, bool hasMipmaps /*= false*/, vcTextureWrapMode wrapMode /*= vcTWM_Repeat*/, vcTextureCreationFlags flags /*= vcTCF_None*/, int32_t aniFilter /*= 0*/)
{
  if (ppTexture == nullptr || pFilename == nullptr)
    return false;

  void *pFileData;
  int64_t fileLen;

  if (udFile_Load(pFilename, &pFileData, &fileLen) != udR_Success)
    return false;

  bool result = vcTexture_CreateFromMemory(ppTexture, pFileData, fileLen, pWidth, pHeight, filterMode, hasMipmaps, wrapMode, flags, aniFilter);

  udFree(pFileData);
  return result;
}

udResult vcTexture_UploadPixels(vcTexture *pTexture, const void *pPixels, int width, int height)
{
  if (pTexture == nullptr || pPixels == nullptr || width == 0 || height == 0)
    return udR_InvalidParameter_;

  if (pTexture->format == vcTextureFormat_Unknown || pTexture->format == vcTextureFormat_Cubemap || pTexture->format == vcTextureFormat_Count)
    return udR_InvalidParameter_;

  udResult result = udR_Success;

  GLint textureFormat = GL_INVALID_ENUM;
  GLenum pixelType = GL_INVALID_ENUM;
  GLint pixelFormat = GL_INVALID_ENUM;
  int pixelBytes = 0;
  vcTexture_GetFormatAndPixelSize(pTexture->format, &pixelBytes, &textureFormat, &pixelType, &pixelFormat);

  glBindTexture(GL_TEXTURE_2D, pTexture->id);
  glTexImage2D(GL_TEXTURE_2D, 0, textureFormat, pTexture->width, pTexture->height, 0, pixelFormat, pixelType, pPixels);
  glBindTexture(GL_TEXTURE_2D, 0);

  pTexture->width = width;
  pTexture->height = height;
  vcGLState_ReportGPUWork(0, 0, pTexture->width * pTexture->height * pixelBytes);

epilogue:
  VERIFY_GL();
  return result;
}

void vcTexture_Destroy(vcTexture **ppTexture)
{
  if (ppTexture == nullptr || *ppTexture == nullptr)
    return;

  glDeleteTextures(1, &(*ppTexture)->id);
  glDeleteBuffers(2, (*ppTexture)->pbos);
  udFree(*ppTexture);
  *ppTexture = nullptr;

  VERIFY_GL();
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
  int pixelBytes = 4;

  for (int i = 0; i < 6; i++) // for each face of the cube map
  {
    int width, height, depth;

    char fileNameNoExt[256] = "";
    fileName.ExtractFilenameOnly(fileNameNoExt, (int)udLengthOf(fileNameNoExt));
    uint8_t* data = stbi_load(vcSettings_GetAssetPath(udTempStr("assets/skyboxes/%s%s%s", fileNameNoExt, names[i], fileName.GetExt())), &width, &height, &depth, 0);

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

    pixelBytes = depth; // assume they all have the same depth
  }

  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

  if (pTexture->id == GL_INVALID_INDEX)
  {
    udFree(pTexture);
    return false;
  }

  vcGLState_ReportGPUWork(0, 0, pTexture->width * pTexture->height * pixelBytes * 6);

  *ppTexture = pTexture;
  VERIFY_GL();
  return true;
}

udResult vcTexture_GetSize(vcTexture *pTexture, int *pWidth, int *pHeight)
{
  if (pTexture == nullptr)
    return udR_InvalidParameter_;

  if (pWidth != nullptr)
    *pWidth = (int)pTexture->width;

  if (pHeight != nullptr)
    *pHeight = (int)pTexture->height;

  return udR_Success;
}

bool vcTexture_BeginReadPixels(vcTexture *pTexture, uint32_t x, uint32_t y, uint32_t width, uint32_t height, void *pPixels, vcFramebuffer *pFramebuffer /*= nullptr*/)
{
  if (pFramebuffer == nullptr || pTexture == nullptr || pPixels == nullptr || int(x + width) > pTexture->width || int(y + height) > pTexture->height)
    return false;

  if (pTexture->format == vcTextureFormat_Unknown || pTexture->format == vcTextureFormat_Cubemap || pTexture->format == vcTextureFormat_Count)
    return false;

  udResult result = udR_Success;
  void *pPixelBuffer = pPixels;
  GLenum pixelType = GL_INVALID_ENUM;
  GLint pixelFormat = GL_INVALID_ENUM;
  vcTexture_GetFormatAndPixelSize(pTexture->format, nullptr, nullptr, &pixelType, &pixelFormat);

  UD_ERROR_IF(!vcFramebuffer_Bind(pFramebuffer), udR_InternalError);

  // Only asychronously transfer if texture is configured for it
  if ((pTexture->flags & vcTCF_AsynchronousRead) == vcTCF_AsynchronousRead)
  {
    // Copy to PBO
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pTexture->pbos[pTexture->pboIndex]);
    pPixelBuffer = nullptr;
  }

  glReadPixels(x, y, width, height, pixelFormat , pixelType, pPixelBuffer);

  if ((pTexture->flags & vcTCF_AsynchronousRead) == vcTCF_AsynchronousRead)
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

epilogue:
  VERIFY_GL();
  return result == udR_Success;
}

bool vcTexture_EndReadPixels(vcTexture *pTexture, uint32_t x, uint32_t y, uint32_t width, uint32_t height, void *pPixels)
{
  if (pTexture == nullptr || pPixels == nullptr || int(x + width) > pTexture->width || int(y + height) > pTexture->height)
    return false;

  if (pTexture->format == vcTextureFormat_Unknown || pTexture->format == vcTextureFormat_Cubemap || pTexture->format == vcTextureFormat_Count || (pTexture->flags & vcTCF_AsynchronousRead) != vcTCF_AsynchronousRead)
    return false;

  udResult result = udR_Success;
  int pixelBytes = 0;
  vcTexture_GetFormatAndPixelSize(pTexture->format, &pixelBytes);

  // Read previous PBO back to CPU
  glBindBuffer(GL_PIXEL_PACK_BUFFER, pTexture->pbos[pTexture->pboIndex]);
  pTexture->pboIndex = (pTexture->pboIndex + 1) & 1;

  uint8_t *pData = (uint8_t *)glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, width * height * pixelBytes, GL_MAP_READ_BIT);
  if (pData != nullptr)
  {
    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    memcpy(pPixels, pData, width * height * pixelBytes);
  }
  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

epilogue:
  VERIFY_GL();
  return result == udR_Success;
}
