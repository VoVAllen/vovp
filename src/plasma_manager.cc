#include "vovp/utils.h"
#include <vovp/plasma_manager.h>

namespace vovp {

using namespace plasma;
using arrow::cuda::CudaBuffer;

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

void VovpPlasmaManager::Release(std::string &object_id) {
  auto plasma_object_id = ToObjectID(object_id);
  VOVP_CHECK_ARROW(client->Release(plasma_object_id));
}

DLManagedTensor *VovpPlasmaManager::PutDlpackTensor(
    DLManagedTensor *dlm_tensor, std::string &object_id,
    bool release_when_destruct, bool try_delete_when_destruct,
    bool try_delete_before_create) {
  CHECK(IsContiguous(dlm_tensor));

  auto plasma_object_id = ToObjectID(object_id);
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
    auto result = CudaBuffer::FromBuffer(buffer);
    if (result.ok()) {
      auto gpu_buffer = result.ValueOrDie();
      auto result = gpu_buffer->CopyFromDevice(0, dl_tensor->data, data_size);
      check_arrow_status(result);
    } else {
      LOG(FATAL) << "Invalid buffer";
    }
  }

  auto plasma_dlm_tensor = CreatePlasmaBufferToDlpack(
      dlm_tensor, buffer, client, plasma_object_id, try_delete_when_destruct,
      release_when_destruct);
  check_arrow_status(client->Seal(plasma_object_id));

  return plasma_dlm_tensor;
}

DLManagedTensor *VovpPlasmaManager::GetDlpackTensor(std::string &object_id) {
  auto plasma_object_id = ToObjectID(object_id);
  std::vector<ObjectBuffer> obj_buffers;
  std::vector<ObjectID> object_ids = {plasma_object_id};
  auto status = client->Get(object_ids, 1000, &obj_buffers);
  check_arrow_status(status);
  CHECK(obj_buffers[0].data) << "Unable to get tensor " << object_id;
  auto dlm_tensor = GetPlasmaBufferToDlpack(
      obj_buffers[0].data, obj_buffers[0].metadata, client, plasma_object_id);
  return dlm_tensor;
}

} // namespace vovp
