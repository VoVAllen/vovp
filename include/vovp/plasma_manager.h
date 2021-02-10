#ifndef VOVP_PLASMA_MANAGER_H
#define VOVP_PLASMA_MANAGER_H
#include <arrow/gpu/cuda_memory.h>
#include <arrow/io/memory.h>
#include <dlpack/dlpack.h>
#include <dmlc/base.h>
#include <dmlc/io.h>
#include <dmlc/logging.h>
#include <dmlc/memory_io.h>
#include <memory>
#include <plasma/client.h>
#include <plasma/common.h>
#include <random>
#include <vovp/ndarray_utils.h>
#include <vovp/serializer.h>
#include <vovp/utils.h>

namespace vovp {

using namespace plasma;

ObjectID random_object_id();
class VovpPlasmaManager {
public:
  VovpPlasmaManager(std::string socket_name);
  // ~VovpPlasmaManager();
  DLManagedTensor *PutDlpackTensor(DLManagedTensor *dlm_tensor,
                                     std::string& object_id, bool release_when_destruct=true,bool try_delete_when_destruct=false, bool try_delete_before_create=true);

  DLManagedTensor *GetDlpackTensor(std::string& object_id);

  void Release(std::string& object_id);

  std::shared_ptr<PlasmaClient> client;
  ObjectID tmp_object_id;
};
} // namespace vovp
#endif /* VOVP_PLASMA_MANAGER_H */
