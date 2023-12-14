

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "edge-impulse-sdk/tensorflow/lite/c/builtin_op_data.h"
#include "edge-impulse-sdk/tensorflow/lite/c/common.h"
#include "edge-impulse-sdk/tensorflow/lite/micro/kernels/micro_ops.h"

#if defined __GNUC__
#define ALIGN(X) __attribute__((aligned(X)))
#elif defined _MSC_VER
#define ALIGN(X) __declspec(align(X))
#elif defined __TASKING__
#define ALIGN(X) __align(X)
#endif

namespace {

constexpr int kTensorArenaSize = 144;

#if defined(EI_CLASSIFIER_ALLOCATION_STATIC)
uint8_t tensor_arena[kTensorArenaSize] ALIGN(16);
#elif defined(EI_CLASSIFIER_ALLOCATION_STATIC_HIMAX)
#pragma Bss(".tensor_arena")
uint8_t tensor_arena[kTensorArenaSize] ALIGN(16);
#pragma Bss()
#else
#define EI_CLASSIFIER_ALLOCATION_HEAP 1
uint8_t* tensor_arena = NULL;
#endif

static uint8_t* tensor_boundary;
static uint8_t* current_location;

template <int SZ, class T> struct TfArray {
  int sz; T elem[SZ];
};
enum used_operators_e {
  OP_FULLY_CONNECTED, OP_SOFTMAX,  OP_LAST
};
struct TensorInfo_t { // subset of TfLiteTensor used for initialization from constant memory
  TfLiteAllocationType allocation_type;
  TfLiteType type;
  void* data;
  TfLiteIntArray* dims;
  size_t bytes;
  TfLiteQuantization quantization;
};
struct NodeInfo_t { // subset of TfLiteNode used for initialization from constant memory
  struct TfLiteIntArray* inputs;
  struct TfLiteIntArray* outputs;
  void* builtin_data;
  used_operators_e used_op_index;
};

TfLiteContext ctx{};
TfLiteTensor tflTensors[11];
TfLiteRegistration registrations[OP_LAST];
TfLiteNode tflNodes[4];

const TfArray<2, int> tensor_dimension0 = { 2, { 1,33 } };
const TfArray<1, float> quant0_scale = { 1, { 0.44246551394462585, } };
const TfArray<1, int> quant0_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant0 = { (TfLiteFloatArray*)&quant0_scale, (TfLiteIntArray*)&quant0_zero, 0 };
const ALIGN(8) int32_t tensor_data1[20] = { 0, 16, -12, 0, 8, -17, 0, 6, 29, -38, 0, -103, -31, -46, -21, -30, -16, 0, -33, 12, };
const TfArray<1, int> tensor_dimension1 = { 1, { 20 } };
const TfArray<1, float> quant1_scale = { 1, { 0.0019613932818174362, } };
const TfArray<1, int> quant1_zero = { 1, { 0 } };
const TfLiteAffineQuantization quant1 = { (TfLiteFloatArray*)&quant1_scale, (TfLiteIntArray*)&quant1_zero, 0 };
const ALIGN(8) int32_t tensor_data2[10] = { -96, -18, 118, -36, 32, -171, 148, 127, -89, 120, };
const TfArray<1, int> tensor_dimension2 = { 1, { 10 } };
const TfArray<1, float> quant2_scale = { 1, { 0.00064352090703323483, } };
const TfArray<1, int> quant2_zero = { 1, { 0 } };
const TfLiteAffineQuantization quant2 = { (TfLiteFloatArray*)&quant2_scale, (TfLiteIntArray*)&quant2_zero, 0 };
const ALIGN(8) int32_t tensor_data3[3] = { 168, 49, -151, };
const TfArray<1, int> tensor_dimension3 = { 1, { 3 } };
const TfArray<1, float> quant3_scale = { 1, { 0.00075003970414400101, } };
const TfArray<1, int> quant3_zero = { 1, { 0 } };
const TfLiteAffineQuantization quant3 = { (TfLiteFloatArray*)&quant3_scale, (TfLiteIntArray*)&quant3_zero, 0 };
const ALIGN(8) int8_t tensor_data4[20*33] = { 
  -56, -52, -45, 9, -12, 6, -64, -55, -60, 11, 68, -37, -51, 66, 31, 73, 68, -24, 51, -48, 65, -68, 70, -55, -43, -44, -68, 31, -29, -26, -47, -66, 14, 
  2, 19, -6, -27, 7, 71, 52, 62, -61, 21, 3, -73, -29, -79, -57, -45, -42, -78, 52, 34, -27, -62, 19, -56, 20, -57, 14, 25, -44, 4, 60, 65, -76, 
  -5, -79, -64, 27, -5, 12, -77, -18, -47, 3, -30, -24, -55, -16, 40, -63, -51, -18, 62, -72, 51, -11, -31, 15, -34, 63, 8, -39, -12, -45, -49, 56, -65, 
  -73, -20, -60, 31, 20, 22, -52, 12, 11, 64, 0, -65, 12, 33, 6, -37, -20, -16, 71, 44, -75, -47, 75, -6, -10, 69, 5, -25, 57, 14, 0, 55, 5, 
  2, 36, -38, 50, 25, 66, -32, -29, 14, 5, 14, -42, -5, -16, -64, -3, -37, -33, 17, 14, 2, -41, 68, 30, 11, -23, 84, 45, 88, -1, 5, -21, -18, 
  52, -47, 1, -3, 4, 28, 14, 25, -1, -38, 10, -14, 87, -16, 74, 33, -20, 46, 51, 63, -16, -49, -82, -23, -54, 38, 10, -30, 38, 25, 34, 69, 1, 
  -56, 68, -9, -27, -29, 22, 22, -17, -35, 40, -62, -57, -75, -53, 65, 18, -16, 36, 20, -27, 14, -9, 69, -57, 19, 1, -37, 12, 19, -21, 33, -22, -47, 
  -18, 80, -49, -22, -58, 70, 2, -70, 60, -34, 56, -9, -54, -33, -42, -8, -1, -59, 4, -21, -23, -60, -121, -15, -12, -65, 52, 23, 27, 52, -66, 63, -66, 
  29, 45, 27, -60, 60, 37, -9, -15, 31, -43, -69, 116, 24, -10, -57, -27, 31, 6, 54, -1, 12, 0, -44, -41, -41, 64, -74, 13, 50, -1, -26, 19, -42, 
  -20, -2, -10, 20, -80, -20, -16, -28, 38, -26, -14, -16, -36, -3, 6, -32, 20, 66, 61, -61, 47, -48, -16, 63, 57, -56, -26, -47, 65, 43, -58, 23, -2, 
  -60, -62, -71, -56, 71, -25, 24, -2, 10, 51, 6, 64, 14, 63, -9, -55, -47, 46, -23, -8, -41, 4, 47, -68, -33, 24, -54, 42, 26, -62, 72, -13, -70, 
  44, 2, -21, -46, -9, 41, -15, 52, 58, 93, 23, 85, -1, -61, -55, 22, 1, 33, 58, 55, 30, -36, -92, 9, 43, -58, -25, 51, -57, 46, -66, 21, -58, 
  29, 25, 17, -29, 36, -22, -4, -33, -11, 55, 2, -75, -78, 37, 42, -97, 14, -29, -72, -16, -99, -44, 127, -2, -49, 3, -42, 34, 72, -32, 50, -11, -10, 
  -1, -1, -25, 38, -4, -28, 20, -15, 23, 45, 81, 35, 23, -46, -10, 1, -58, -54, 20, 61, -89, -44, 21, -33, -60, 63, 55, 16, 79, -33, -74, 90, 36, 
  -50, -85, 70, -44, 28, -30, 29, 46, 7, -35, -60, -22, -65, -25, 1, -17, -47, -39, -32, -44, 6, 21, -57, 14, 32, -7, -67, -21, -42, 5, 12, -13, 43, 
  -59, 60, -35, 13, -33, 52, -57, -37, -10, 44, -49, 46, -25, -14, 83, -50, 3, 58, 46, 52, 25, -26, -12, -60, -34, -22, 45, 27, 7, 7, -69, -61, -66, 
  -66, 55, 35, 4, -13, -9, -18, -36, -9, 14, -6, -61, -43, -67, -49, 36, 52, 36, -63, -9, -35, -68, -43, 47, 39, 19, -24, -1, 67, 70, 5, -19, -12, 
  -33, 4, -71, -64, 66, -32, 13, 53, -43, 6, 7, 32, 16, -23, -12, -22, 3, -12, -40, 60, -68, -21, 42, -61, 66, -9, -53, 73, 44, 39, 44, 13, 13, 
  -45, -32, 66, 68, 40, 20, -22, -51, -20, -24, -34, -10, 81, 17, -5, 77, -3, 13, -62, -13, 76, -36, -84, 52, -66, -81, 38, -39, -4, -5, 42, 11, 43, 
  -31, 40, 42, 23, 18, -3, 10, -64, 66, 62, -48, 43, -11, -29, 57, 18, 12, -76, 21, -11, -28, 37, 116, -78, 47, 57, -6, 61, -71, -74, -71, -57, 72, 
};
const TfArray<2, int> tensor_dimension4 = { 2, { 20,33 } };
const TfArray<1, float> quant4_scale = { 1, { 0.0044328728690743446, } };
const TfArray<1, int> quant4_zero = { 1, { 0 } };
const TfLiteAffineQuantization quant4 = { (TfLiteFloatArray*)&quant4_scale, (TfLiteIntArray*)&quant4_zero, 0 };
const ALIGN(8) int8_t tensor_data5[10*20] = { 
  -63, -18, 23, 17, 81, -73, -27, 17, -98, -95, -31, -69, 75, 66, -35, -7, -89, 14, -42, -3, 
  -64, 65, -50, 38, 25, 41, 69, -74, -13, -72, 16, -52, 68, -37, -76, -85, -41, 65, 38, -119, 
  -77, 70, 24, -50, 61, -10, 41, -48, 67, 75, -58, -54, 28, -127, 27, -42, -23, 97, -70, -25, 
  -65, -16, -66, -81, -6, 56, 63, 55, 91, -60, 76, 48, 9, -39, -22, 95, 10, 28, 24, -61, 
  15, 45, -59, -75, -47, 111, -35, 55, -55, 88, 5, -63, -107, 13, -15, 7, 18, -88, 114, -1, 
  60, 83, -86, 9, -74, -9, 3, -72, 83, -78, -102, 101, -80, 60, -62, 47, -97, -8, -81, 56, 
  78, -38, 18, 78, 93, -37, 12, 74, 49, 68, -48, 50, -105, 55, 12, 13, 39, -21, 20, -49, 
  -54, 81, -54, -94, -20, -32, -29, -36, 92, 51, 63, 80, -21, -20, -12, -14, -47, 16, -51, 49, 
  -67, 63, 1, 46, 71, -11, 56, -106, -36, -83, 41, 62, 76, 67, -34, 32, -69, 89, 0, -10, 
  -27, 27, 21, 76, -71, 45, 70, -9, 79, -48, -19, 20, -86, -49, -71, 43, 107, 92, -39, 95, 
};
const TfArray<2, int> tensor_dimension5 = { 2, { 10,20 } };
const TfArray<1, float> quant5_scale = { 1, { 0.0043523777276277542, } };
const TfArray<1, int> quant5_zero = { 1, { 0 } };
const TfLiteAffineQuantization quant5 = { (TfLiteFloatArray*)&quant5_scale, (TfLiteIntArray*)&quant5_zero, 0 };
const ALIGN(8) int8_t tensor_data6[3*10] = { 
  -19, -127, -69, 76, 62, 7, 41, 34, -55, -3, 
  44, -3, 76, -59, 17, -4, -7, 29, 53, 5, 
  -86, -6, -2, 46, 54, 21, -21, -5, 20, -39, 
};
const TfArray<2, int> tensor_dimension6 = { 2, { 3,10 } };
const TfArray<1, float> quant6_scale = { 1, { 0.0077296607196331024, } };
const TfArray<1, int> quant6_zero = { 1, { 0 } };
const TfLiteAffineQuantization quant6 = { (TfLiteFloatArray*)&quant6_scale, (TfLiteIntArray*)&quant6_zero, 0 };
const TfArray<2, int> tensor_dimension7 = { 2, { 1,20 } };
const TfArray<1, float> quant7_scale = { 1, { 0.14785502851009369, } };
const TfArray<1, int> quant7_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant7 = { (TfLiteFloatArray*)&quant7_scale, (TfLiteIntArray*)&quant7_zero, 0 };
const TfArray<2, int> tensor_dimension8 = { 2, { 1,10 } };
const TfArray<1, float> quant8_scale = { 1, { 0.097033977508544922, } };
const TfArray<1, int> quant8_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant8 = { (TfLiteFloatArray*)&quant8_scale, (TfLiteIntArray*)&quant8_zero, 0 };
const TfArray<2, int> tensor_dimension9 = { 2, { 1,3 } };
const TfArray<1, float> quant9_scale = { 1, { 0.078951217234134674, } };
const TfArray<1, int> quant9_zero = { 1, { -64 } };
const TfLiteAffineQuantization quant9 = { (TfLiteFloatArray*)&quant9_scale, (TfLiteIntArray*)&quant9_zero, 0 };
const TfArray<2, int> tensor_dimension10 = { 2, { 1,3 } };
const TfArray<1, float> quant10_scale = { 1, { 0.00390625, } };
const TfArray<1, int> quant10_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant10 = { (TfLiteFloatArray*)&quant10_scale, (TfLiteIntArray*)&quant10_zero, 0 };
const TfLiteFullyConnectedParams opdata0 = { kTfLiteActRelu, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
const TfArray<3, int> inputs0 = { 3, { 0,4,1 } };
const TfArray<1, int> outputs0 = { 1, { 7 } };
const TfLiteFullyConnectedParams opdata1 = { kTfLiteActRelu, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
const TfArray<3, int> inputs1 = { 3, { 7,5,2 } };
const TfArray<1, int> outputs1 = { 1, { 8 } };
const TfLiteFullyConnectedParams opdata2 = { kTfLiteActNone, kTfLiteFullyConnectedWeightsFormatDefault, false, false };
const TfArray<3, int> inputs2 = { 3, { 8,6,3 } };
const TfArray<1, int> outputs2 = { 1, { 9 } };
const TfLiteSoftmaxParams opdata3 = { 1 };
const TfArray<1, int> inputs3 = { 1, { 9 } };
const TfArray<1, int> outputs3 = { 1, { 10 } };
const TensorInfo_t tensorData[] = {
  { kTfLiteArenaRw, kTfLiteInt8, tensor_arena + 0, (TfLiteIntArray*)&tensor_dimension0, 33, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant0))}, },
  { kTfLiteMmapRo, kTfLiteInt32, (void*)tensor_data1, (TfLiteIntArray*)&tensor_dimension1, 80, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant1))}, },
  { kTfLiteMmapRo, kTfLiteInt32, (void*)tensor_data2, (TfLiteIntArray*)&tensor_dimension2, 40, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant2))}, },
  { kTfLiteMmapRo, kTfLiteInt32, (void*)tensor_data3, (TfLiteIntArray*)&tensor_dimension3, 12, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant3))}, },
  { kTfLiteMmapRo, kTfLiteInt8, (void*)tensor_data4, (TfLiteIntArray*)&tensor_dimension4, 660, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant4))}, },
  { kTfLiteMmapRo, kTfLiteInt8, (void*)tensor_data5, (TfLiteIntArray*)&tensor_dimension5, 200, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant5))}, },
  { kTfLiteMmapRo, kTfLiteInt8, (void*)tensor_data6, (TfLiteIntArray*)&tensor_dimension6, 30, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant6))}, },
  { kTfLiteArenaRw, kTfLiteInt8, tensor_arena + 48, (TfLiteIntArray*)&tensor_dimension7, 20, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant7))}, },
  { kTfLiteArenaRw, kTfLiteInt8, tensor_arena + 0, (TfLiteIntArray*)&tensor_dimension8, 10, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant8))}, },
  { kTfLiteArenaRw, kTfLiteInt8, tensor_arena + 16, (TfLiteIntArray*)&tensor_dimension9, 3, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant9))}, },
  { kTfLiteArenaRw, kTfLiteInt8, tensor_arena + 0, (TfLiteIntArray*)&tensor_dimension10, 3, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant10))}, },
};const NodeInfo_t nodeData[] = {
  { (TfLiteIntArray*)&inputs0, (TfLiteIntArray*)&outputs0, const_cast<void*>(static_cast<const void*>(&opdata0)), OP_FULLY_CONNECTED, },
  { (TfLiteIntArray*)&inputs1, (TfLiteIntArray*)&outputs1, const_cast<void*>(static_cast<const void*>(&opdata1)), OP_FULLY_CONNECTED, },
  { (TfLiteIntArray*)&inputs2, (TfLiteIntArray*)&outputs2, const_cast<void*>(static_cast<const void*>(&opdata2)), OP_FULLY_CONNECTED, },
  { (TfLiteIntArray*)&inputs3, (TfLiteIntArray*)&outputs3, const_cast<void*>(static_cast<const void*>(&opdata3)), OP_SOFTMAX, },
};
static std::vector<void*> overflow_buffers;
static TfLiteStatus AllocatePersistentBuffer(struct TfLiteContext* ctx,
                                                 size_t bytes, void** ptr) {
  if (current_location - bytes < tensor_boundary) {
    // OK, this will look super weird, but.... we have CMSIS-NN buffers which
    // we cannot calculate beforehand easily.
    *ptr = malloc(bytes);
    if (*ptr == NULL) {
      printf("ERR: Failed to allocate persistent buffer of size %d\n", (int)bytes);
      return kTfLiteError;
    }
    overflow_buffers.push_back(*ptr);
    return kTfLiteOk;
  }

  current_location -= bytes;

  *ptr = current_location;
  return kTfLiteOk;
}
typedef struct {
  size_t bytes;
  void *ptr;
} scratch_buffer_t;
static std::vector<scratch_buffer_t> scratch_buffers;

static TfLiteStatus RequestScratchBufferInArena(struct TfLiteContext* ctx, size_t bytes,
                                                int* buffer_idx) {
  scratch_buffer_t b;
  b.bytes = bytes;

  TfLiteStatus s = AllocatePersistentBuffer(ctx, b.bytes, &b.ptr);
  if (s != kTfLiteOk) {
    return s;
  }

  scratch_buffers.push_back(b);

  *buffer_idx = scratch_buffers.size() - 1;

  return kTfLiteOk;
}

static void* GetScratchBuffer(struct TfLiteContext* ctx, int buffer_idx) {
  if (buffer_idx > static_cast<int>(scratch_buffers.size()) - 1) {
    return NULL;
  }
  return scratch_buffers[buffer_idx].ptr;
}
} // namespace

  TfLiteStatus trained_model_init( void*(*alloc_fnc)(size_t,size_t) ) {
#ifdef EI_CLASSIFIER_ALLOCATION_HEAP
  tensor_arena = (uint8_t*) alloc_fnc(16, kTensorArenaSize);
  if (!tensor_arena) {
    printf("ERR: failed to allocate tensor arena\n");
    return kTfLiteError;
  }
#endif
  tensor_boundary = tensor_arena;
  current_location = tensor_arena + kTensorArenaSize;
  ctx.AllocatePersistentBuffer = &AllocatePersistentBuffer;
  ctx.RequestScratchBufferInArena = &RequestScratchBufferInArena;
  ctx.GetScratchBuffer = &GetScratchBuffer;
  ctx.tensors = tflTensors;
  ctx.tensors_size = 11;
  for(size_t i = 0; i < 11; ++i) {
    tflTensors[i].type = tensorData[i].type;
    tflTensors[i].is_variable = 0;

#if defined(EI_CLASSIFIER_ALLOCATION_HEAP)
    tflTensors[i].allocation_type = tensorData[i].allocation_type;
#else
    tflTensors[i].allocation_type = (tensor_arena <= tensorData[i].data && tensorData[i].data < tensor_arena + kTensorArenaSize) ? kTfLiteArenaRw : kTfLiteMmapRo;
#endif
    tflTensors[i].bytes = tensorData[i].bytes;
    tflTensors[i].dims = tensorData[i].dims;

#if defined(EI_CLASSIFIER_ALLOCATION_HEAP)
    if(tflTensors[i].allocation_type == kTfLiteArenaRw){
      uint8_t* start = (uint8_t*) ((uintptr_t)tensorData[i].data + (uintptr_t) tensor_arena);

     tflTensors[i].data.data =  start;
    }
    else{
       tflTensors[i].data.data = tensorData[i].data;
    }
#else
    tflTensors[i].data.data = tensorData[i].data;
#endif // EI_CLASSIFIER_ALLOCATION_HEAP
    tflTensors[i].quantization = tensorData[i].quantization;
    if (tflTensors[i].quantization.type == kTfLiteAffineQuantization) {
      TfLiteAffineQuantization const* quant = ((TfLiteAffineQuantization const*)(tensorData[i].quantization.params));
      tflTensors[i].params.scale = quant->scale->data[0];
      tflTensors[i].params.zero_point = quant->zero_point->data[0];
    }
    if (tflTensors[i].allocation_type == kTfLiteArenaRw) {
      auto data_end_ptr = (uint8_t*)tflTensors[i].data.data + tensorData[i].bytes;
      if (data_end_ptr > tensor_boundary) {
        tensor_boundary = data_end_ptr;
      }
    }
  }
  if (tensor_boundary > current_location /* end of arena size */) {
    printf("ERR: tensor arena is too small, does not fit model - even without scratch buffers\n");
    return kTfLiteError;
  }
  registrations[OP_FULLY_CONNECTED] = *tflite::ops::micro::Register_FULLY_CONNECTED();
  registrations[OP_SOFTMAX] = *tflite::ops::micro::Register_SOFTMAX();

  for(size_t i = 0; i < 4; ++i) {
    tflNodes[i].inputs = nodeData[i].inputs;
    tflNodes[i].outputs = nodeData[i].outputs;
    tflNodes[i].builtin_data = nodeData[i].builtin_data;
    tflNodes[i].custom_initial_data = nullptr;
    tflNodes[i].custom_initial_data_size = 0;
    if (registrations[nodeData[i].used_op_index].init) {
      tflNodes[i].user_data = registrations[nodeData[i].used_op_index].init(&ctx, (const char*)tflNodes[i].builtin_data, 0);
    }
  }
  for(size_t i = 0; i < 4; ++i) {
    if (registrations[nodeData[i].used_op_index].prepare) {
      TfLiteStatus status = registrations[nodeData[i].used_op_index].prepare(&ctx, &tflNodes[i]);
      if (status != kTfLiteOk) {
        return status;
      }
    }
  }
  return kTfLiteOk;
}

static const int inTensorIndices[] = {
  0, 
};
TfLiteTensor* trained_model_input(int index) {
  return &ctx.tensors[inTensorIndices[index]];
}

static const int outTensorIndices[] = {
  10, 
};
TfLiteTensor* trained_model_output(int index) {
  return &ctx.tensors[outTensorIndices[index]];
}

TfLiteStatus trained_model_invoke() {
  for(size_t i = 0; i < 4; ++i) {
    TfLiteStatus status = registrations[nodeData[i].used_op_index].invoke(&ctx, &tflNodes[i]);
    if (status != kTfLiteOk) {
      return status;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus trained_model_reset( void (*free_fnc)(void* ptr) ) {
#ifdef EI_CLASSIFIER_ALLOCATION_HEAP
  free_fnc(tensor_arena);
#endif
  scratch_buffers.clear();
  for (size_t ix = 0; ix < overflow_buffers.size(); ix++) {
    free(overflow_buffers[ix]);
  }
  overflow_buffers.clear();
  return kTfLiteOk;
}
