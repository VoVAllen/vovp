#ifndef VOVP_NDARRAY_UTILS_H
#define VOVP_NDARRAY_UTILS_H

#include "vovp/utils.h"
#include <cstdint>
#include <dlpack/dlpack.h>
#include <dmlc/logging.h>
#include <plasma/client.h>
#include <plasma/common.h>

namespace vovp {
using namespace plasma;

bool IsContiguous(DLManagedTensor *data_);

void PlasmaTensorCtxNoReleaseDeleter(DLManagedTensor *arg);
void PlasmaTensorCtxReleaseDeleter(DLManagedTensor *arg);

class BufferDeleter {
public:
  BufferDeleter(std::shared_ptr<PlasmaClient> client, ObjectID object_id)
      : client(client), object_id(object_id){};
  ~BufferDeleter() {

    ObjectTable table;
    VOVP_CHECK_ARROW(client->List(&table));
    // LOG(INFO) << "Table size: " << table.size();
    // for (const auto &kv : table) {
    //   LOG(INFO) << "DL Ref: " << kv.first.hex() << ": " << kv.second->ref_count;
    // }
    // auto status = owner->plasma_client->Delete(owner->object_id);
    // LOG(INFO) << "Delete: " << owner->object_id.hex();
    // check_arrow_status(status);
    // LOG(INFO) << "buf Delete: " << object_id.hex();
    VOVP_CHECK_ARROW(client->Delete(object_id));
  };
  std::shared_ptr<PlasmaClient> client;
  ObjectID object_id;
};

typedef struct PlasmaTensorCtx {
  std::shared_ptr<PlasmaClient> plasma_client;
  // To be executed after destruction of buffer and before
  // destruction of PlasmaClient
  std::unique_ptr<BufferDeleter> buffer_deleter;
  std::shared_ptr<Buffer> buffer;
  ObjectID object_id;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;

  DLManagedTensor tensor;

  PlasmaTensorCtx(std::shared_ptr<Buffer> buffer,
                  std::shared_ptr<PlasmaClient> plasma_client,
                  ObjectID object_id, bool release_when_destruct,
                  bool try_delete_when_destruct)
      : buffer(buffer), plasma_client(plasma_client), object_id(object_id) {
    if (try_delete_when_destruct) {
      buffer_deleter = std::unique_ptr<BufferDeleter>(
          new BufferDeleter(plasma_client, object_id));
    }
    tensor.manager_ctx = this;
    tensor.dl_tensor.dtype.lanes = 1;
    if (release_when_destruct) {
      tensor.deleter = &PlasmaTensorCtxReleaseDeleter;
    } else {
      tensor.deleter = &PlasmaTensorCtxNoReleaseDeleter;
    }
  }
} PlasmaTensorCtx;

DLManagedTensor *GetPlasmaBufferToDlpack(std::shared_ptr<Buffer> buffer,
                                         std::shared_ptr<Buffer> metadatabuffer,
                                         std::shared_ptr<PlasmaClient> client,
                                         ObjectID object_id);

DLManagedTensor *CreatePlasmaBufferToDlpack(
    DLManagedTensor *dlm_tensor, std::shared_ptr<Buffer> buffer,
    std::shared_ptr<PlasmaClient> client, ObjectID object_id,
    bool try_delete_when_destruct);

void *GetPointerFromBuffer(std::shared_ptr<Buffer> buffer,
                           DLDeviceType device);
} // namespace vovp

#endif /* VOVP_NDARRAY_UTILS_H */
