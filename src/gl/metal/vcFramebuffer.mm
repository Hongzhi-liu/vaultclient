#import "gl/vcFramebuffer.h"
#import "vcMetal.h"

uint32_t g_blitTargetID = 0;

bool vcFramebuffer_Create(vcFramebuffer **ppFramebuffer, vcTexture *pTexture, vcTexture *pDepth /*= nullptr*/, int level /*= 0*/)
{
  if (ppFramebuffer == nullptr)
    return false;

  udUnused(level);
  vcFramebuffer *pFramebuffer = udAllocType(vcFramebuffer, 1, udAF_Zero);

  pFramebuffer->pColor = pTexture;
  pFramebuffer->pDepth = pDepth;
  pFramebuffer->clear = 0;
  
  @autoreleasepool
  {
    [_renderer addFramebuffer:pFramebuffer];
  }
  
  *ppFramebuffer = pFramebuffer;
  pFramebuffer = nullptr;

  return true;
}

void vcFramebuffer_Destroy(vcFramebuffer **ppFramebuffer)
{
  if (ppFramebuffer == nullptr || *ppFramebuffer == nullptr)
    return;

  @autoreleasepool
  {
    [_renderer destroyFramebuffer:*ppFramebuffer];
  }
  
  udFree(*ppFramebuffer);
  *ppFramebuffer = nullptr;
}

bool vcFramebuffer_Bind(vcFramebuffer *pFramebuffer)
{
  [_renderer setFramebuffer:pFramebuffer];

  return true;
}

bool vcFramebuffer_Clear(vcFramebuffer *pFramebuffer, uint32_t colour)
{
  if (pFramebuffer == nullptr)
    return false;

  if (pFramebuffer->clear != colour)
  {
    udFloat4 col = udFloat4::create(((colour >> 16) & 0xFF) / 255.f, ((colour >> 8) & 0xFF) / 255.f, (colour & 0xFF) / 255.f, ((colour >> 24) & 0xFF) / 255.f);

    _renderer.renderPasses[pFramebuffer->ID].colorAttachments[0].clearColor = MTLClearColorMake(col.x,col.y,col.z,col.w);
    _renderer.renderPasses[pFramebuffer->ID].depthAttachment.clearDepth = 1.0;
  }

  return true;
}
