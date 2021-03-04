#include "vovp/utils.h"
#include <vovp/plasma_manager.h>
#ifdef VOVP_CUDA
#include <arrow/gpu/cuda_memory.h>
#endif

namespace vovp {

using namespace plasma;
#ifdef VOVP_CUDA
using arrow::cuda::CudaBuffer;
#endif

ObjectID random_object_id() {
  static uint32_t random_seed = 0;
  std::mt19937 gen(random_seed++);
  std::uniform_int_distribution<uint32_t> d(
      0, std::numeric_limits<uint8_t>::max());
  ObjectID result;
  uint8_t *data = result.mutable_data();
  std::generate(data, data + kUniqueIDSize,
                [&d, &gen] { return static_cast<uint8_t>(d(gen)); });
  return result;
}

VovpPlasmaManager::VovpPlasmaManager(std::string socket_name) {
  client = std::shared_ptr<PlasmaClient>(
      new PlasmaClient(),
      [](PlasmaClient *client) { check_arrow_status(client->Disconnect()); });
  auto status = client->Connect(socket_name, "", 0, 10);
  CHECK(status.ok()) << "Connection failed: " << status.ToString();
}

// VovpPlasmaManager::Release(std::string& object_id);
// VovpPlasmaManager::~VovpPlasmaManager() {
//   // check_arrow_status(client->Disconnect());
// }

// bool IsPlasmaTensor(DLManagedTensor *dlm_tensor){
//   PlasmaTensorCtx *owner = dynamic_cast<PlasmaTensorCtx
//   *>(dlm_tensor->manager_ctx); return owner;
// }

void VovpPlasmaManager::Release(ObjectID &plasma_object_id) {
  // auto plasma_object_id = ToObjectID(object_id);
  VOVP_CHECK_ARROW(client->Release(plasma_object_id));
}

DLManagedTensor *VovpPlasmaManager::PutDlpackTensor(
    DLManagedTensor *dlm_tensor, ObjectID &plasma_object_id, bool try_delete_when_destruct,
    bool try_delete_before_create) {
  CHECK(IsContiguous(dlm_tensor));

  // auto plasma_object_id = ToObjectID(object_id);
  auto dl_tensor = &(dlm_tensor->dl_tensor);
  auto ndim = dlm_tensor->dl_tensor.ndim;
  int64_t data_size = dl_tensor->dtype.bits / 8;
  for (int i = 0; i < ndim; i++) {
    data_size *= dl_tensor->shape[i];
  }
  std::string metadata;
  metadata.reserve(64);
  dmlc::MemoryStringStream strm_(&metadata);
  auto strm = static_cast<dmlc::Stream *>(&strm_);
  strm->Write(dl_tensor->ctx);
  strm->Write(dl_tensor->dtype);
  strm->Write(ndim);
  strm->WriteArray(dl_tensor->shape, ndim);

  std::shared_ptr<Buffer> buffer;
  auto metadata_ptr = reinterpret_cast<const uint8_t *>(metadata.c_str());
  int device_num = 0;
  if (dl_tensor->ctx.device_type == kDLCPU) {
    device_num = 0;
  } else if (dl_tensor->ctx.device_type == kDLGPU) {
    device_num = 1 + dl_tensor->ctx.device_id;
  }

  if (try_delete_before_create) {
    VOVP_CHECK_ARROW(client->Delete(plasma_object_id));
  }
  auto status = client->Create(plasma_object_id, data_size, metadata_ptr,
                               metadata.size(), &buffer, device_num);
  check_arrow_status(status);
  // Copy tensor data to plasma buffer
  if (dl_tensor->ctx.device_type == kDLCPU) {
    arrow::io::FixedSizeBufferWriter writer(std::move(buffer));
    writer.set_memcopy_threads(4);
    auto result = writer.Write(dl_tensor->data, data_size);
    check_arrow_status(result);
  } else if (dl_tensor->ctx.device_type == kDLGPU) {
#ifdef VOVP_CUDA
    auto result = CudaBuffer::FromBuffer(buffer);
    if (result.ok()) {
      auto gpu_buffer = result.ValueOrDie();
      auto result = gpu_buffer->CopyFromDevice(0, dl_tensor->data, data_size);
      check_arrow_status(result);
    } else {
      LOG(FATAL) << "Invalid buffer";
    }
#else
    LOG(FATAL) << "Unsupport CUDA operation";
#endif
  }

  auto plasma_dlm_tensor = CreatePlasmaBufferToDlpack(
      dlm_tensor, buffer, client, plasma_object_id, try_delete_when_destruct);
  check_arrow_status(client->Seal(plasma_object_id));

  return plasma_dlm_tensor;
}

DLManagedTensor *
VovpPlasmaManager::GetDlpackTensor(ObjectID &plasma_object_id) {
  // auto plasma_object_id = ToObjectID(object_id);
  std::vector<ObjectBuffer> obj_buffers;
  std::vector<ObjectID> object_ids = {plasma_object_id};
  VOVP_CHECK_ARROW(client->Get(object_ids, 1000, &obj_buffers));
  CHECK(obj_buffers[0].data)
      << "Unable to get tensor " << plasma_object_id.hex();
  auto dlm_tensor = GetPlasmaBufferToDlpack(
      obj_buffers[0].data, obj_buffers[0].metadata, client, plasma_object_id);
  return dlm_tensor;
}

DLManagedTensor *VovpPlasmaManager::CreateTensor(ObjectID &object_id,
                                                 int64_t *shape, int ndim,
                                                 DLDataType dtype,
                                                 DLContext ctx) {
  int64_t data_size = dtype.bits / 8;
  for (int i = 0; i < ndim; i++) {
    data_size *= shape[i];
  }

  std::string metadata;
  metadata.reserve(64);
  dmlc::MemoryStringStream strm_(&metadata);
  auto strm = static_cast<dmlc::Stream *>(&strm_);
  strm->Write(ctx);
  strm->Write(dtype);
  strm->Write(ndim);
  strm->WriteArray(shape, ndim);

  int device_num = 0;
  if (ctx.device_type == kDLCPU) {
    device_num = 0;
  } else if (ctx.device_type == kDLGPU) {
    device_num = 1 + ctx.device_id;
  }
  const uint8_t *meta_ptr = reinterpret_cast<const uint8_t *>(metadata.c_str());

  std::shared_ptr<Buffer> buffer;
  auto status = client->Create(object_id, data_size, meta_ptr, metadata.size(),
                               &buffer, device_num);

  auto ptensor_ctx =
      new PlasmaTensorCtx(buffer, client, object_id, true, false);
  auto dltensor = &ptensor_ctx->tensor;
  dltensor->dl_tensor.ctx = ctx;
  dltensor->dl_tensor.dtype = dtype;
  dltensor->dl_tensor.ndim = ndim;
  dltensor->dl_tensor.data =
      GetPointerFromBuffer(buffer, dltensor->dl_tensor.ctx.device_type);
  std::vector<int64_t> *shape_arr = &ptensor_ctx->shape;
  ptensor_ctx->shape.assign(shape, shape + ndim);
  std::vector<int64_t> *stride_arr = &ptensor_ctx->strides;
  stride_arr->resize(ndim, 1);
  for (int i = ndim - 2; i >= 0; --i) {
    (*stride_arr)[i] = (*shape_arr)[i + 1] * (*stride_arr)[i + 1];
  }
  dltensor->dl_tensor.strides = &ptensor_ctx->strides[0];
  dltensor->dl_tensor.shape = &ptensor_ctx->shape[0];

  check_arrow_status(client->Seal(object_id));

  return dltensor;
};

} // namespace vovp
