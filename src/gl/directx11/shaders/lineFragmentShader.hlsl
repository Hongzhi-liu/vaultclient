cbuffer u_cameraPlaneParams
{
  float s_CameraNearPlane;
  float s_CameraFarPlane;
  float u_clipZNear;
  float u_clipZFar;
};

//Input Format
struct PS_INPUT
{
  float4 pos : SV_POSITION;
  float4 colour : COLOR0;
  float2 fLogDepth : TEXCOORD0;
};

//Output Format
struct PS_OUTPUT
{
  float4 Color0 : SV_Target;
  float Depth0 : SV_Depth;
};

PS_OUTPUT main(PS_INPUT input)
{
  PS_OUTPUT output;

  output.Color0 = input.colour;

  float halfFcoef = 1.0 / log2(s_CameraFarPlane + 1.0);
  output.Depth0 = log2(input.fLogDepth.x) * halfFcoef;

  output.Color0.a = output.Depth0; // depth packed here
  return output;
}
