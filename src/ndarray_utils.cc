#include "dlpack/dlpack.h"
#include "vovp/utils.h"
#include <dmlc/memory_io.h>
#include <vovp/ndarray_utils.h>

#include <arrow/gpu/cuda_memory.h>
#include <plasma/client.h>
#include <plasma/common.h>
namespace vovp {
using arrow::cuda::CudaBuffer;

bool IsContiguous(DLManagedTensor *data_) {
  CHECK(data_ != nullptr);
  if (data_->dl_tensor.strides == nullptr)
    return true;
  // See https://github.com/dmlc/dgl/issues/2118 and PyTorch's
  // compute_contiguous() implementation
  int64_t z = 1;
  for (int64_t i = data_->dl_tensor.ndim - 1; i >= 0; --i) {
    if (data_->dl_tensor.shape[i] != 1) {
      if (data_->dl_tensor.strides[i] == z)
        z *= data_->dl_tensor.shape[i];
      else
        return false;
    }
  }
  return true;
}

void PlasmaTensorCtxNoReleaseDeleter(DLManagedTensor *arg) {
  PlasmaTensorCtx *owner = static_cast<PlasmaTensorCtx *>(arg->manager_ctx);
  auto status = owner->plasma_client->Release(owner->object_id);
  VOVP_CHECK_ARROW(status);
  delete owner;
}

void PlasmaTensorCtxReleaseDeleter(DLManagedTensor *arg) {
  PlasmaTensorCtx *owner = static_cast<PlasmaTensorCtx *>(arg->manager_ctx);
  delete owner;
}

void *GetPointerFromBuffer(std::shared_ptr<Buffer> buffer,
                           DLDeviceType device) {
  if (device == kDLCPU) {
    return reinterpret_cast<void *>(buffer->address());
  } else if (device == kDLGPU) {
    auto result = CudaBuffer::FromBuffer(buffer);
    if (result.ok()) {
      auto gpu_buffer = result.ValueOrDie();
      return reinterpret_cast<void *>(gpu_buffer->address());
    } else {
      LOG(FATAL) << "Invalid buffer";
      return nullptr;
    }
  } else {
    LOG(FATAL) << "Invalid device";
    return nullptr;
  }
}

DLManagedTensor *GetPlasmaBufferToDlpack(std::shared_ptr<Buffer> buffer,
                                         std::shared_ptr<Buffer> metadatabuffer,
                                         std::shared_ptr<PlasmaClient> client,
                                         ObjectID object_id) {
  auto ptensor_ctx = new PlasmaTensorCtx(buffer, client, object_id, false);
  void *read_ptr;
  std::vector<uint8_t> read_data;
  if (!buffer->is_cpu()) {
    arrow::cuda::CudaBufferReader reader(metadatabuffer);
    read_data.resize(metadatabuffer->size());
    auto status = reader.Read(metadatabuffer->size(), read_data.data());
    read_ptr = read_data.data();
  } else {
    read_ptr = const_cast<uint8_t *>(buffer->data());
  }

  dmlc::MemoryFixedSizeStream strm_(read_ptr, metadatabuffer->size());
  auto strm = static_cast<dmlc::Stream *>(&strm_);
  auto dltensor = &ptensor_ctx->tensor;
  strm->Read(&dltensor->dl_tensor.ctx);
  strm->Read(&dltensor->dl_tensor.dtype);
  int ndim = 0;
  strm->Read(&ndim);
  ptensor_ctx->shape.resize(ndim);
  strm->ReadArray(&ptensor_ctx->shape[0], ndim);

  dltensor->dl_tensor.ndim = ndim;
  std::vector<int64_t> *shape_arr = &ptensor_ctx->shape;
  std::vector<int64_t> *stride_arr = &ptensor_ctx->strides;
  shape_arr->resize(ndim);
  stride_arr->resize(ndim, 1);
  for (int i = ndim - 2; i >= 0; --i) {
    (*stride_arr)[i] = (*shape_arr)[i + 1] * (*stride_arr)[i + 1];
  }
  dltensor->dl_tensor.strides = &ptensor_ctx->strides[0];
  dltensor->dl_tensor.shape = &ptensor_ctx->shape[0];
  dltensor->manager_ctx = ptensor_ctx;
  ptensor_ctx->tensor.dl_tensor.data =
      GetPointerFromBuffer(buffer, dltensor->dl_tensor.ctx.device_type);
  return &ptensor_ctx->tensor;
};

DLManagedTensor *CreatePlasmaBufferToDlpack(
    DLManagedTensor *dlm_tensor, std::shared_ptr<Buffer> buffer,
    std::shared_ptr<PlasmaClient> client, ObjectID object_id) {
  auto ptensor_ctx = new PlasmaTensorCtx(buffer, client, object_id, true);
  ptensor_ctx->tensor.dl_tensor.ctx = dlm_tensor->dl_tensor.ctx;
  ptensor_ctx->tensor.dl_tensor.dtype = dlm_tensor->dl_tensor.dtype;
  ptensor_ctx->tensor.dl_tensor.ndim = dlm_tensor->dl_tensor.ndim;
  ptensor_ctx->tensor.dl_tensor.data =
      GetPointerFromBuffer(buffer, dlm_tensor->dl_tensor.ctx.device_type);
  std::vector<int64_t> *shape_arr = &ptensor_ctx->shape;
  std::vector<int64_t> *stride_arr = &ptensor_ctx->strides;
  int ndim = ptensor_ctx->tensor.dl_tensor.ndim;
  stride_arr->resize(ndim, 1);
  shape_arr->resize(ndim);
  ptensor_ctx->tensor.dl_tensor.strides = &ptensor_ctx->strides[0];
  ptensor_ctx->tensor.dl_tensor.shape = &ptensor_ctx->shape[0];
  for (int i = 0; i < ndim; i++) {
    (*shape_arr)[i] = dlm_tensor->dl_tensor.shape[i];
  }
  for (int i = ndim - 2; i >= 0; --i) {
    (*stride_arr)[i] = (*shape_arr)[i + 1] * (*stride_arr)[i + 1];
  }
  return &ptensor_ctx->tensor;
};
} // namespace vovp
