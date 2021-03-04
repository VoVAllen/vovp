#ifndef VOVP_PLASMA_MANAGER_H
#define VOVP_PLASMA_MANAGER_H
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
  
  DLManagedTensor *PutDlpackTensor(DLManagedTensor *dlm_tensor,
                                   ObjectID &object_id,
                                   bool try_delete_when_destruct = false,
                                   bool try_delete_before_create = true);

  DLManagedTensor *GetDlpackTensor(ObjectID &object_id);

  DLManagedTensor *CreateTensor(ObjectID &object_id, int64_t *shape, int ndim, DLDataType dtype,
                                  DLContext ctx);

  void Release(ObjectID &object_id);

  std::shared_ptr<PlasmaClient> client;
  ObjectID tmp_object_id;
};
} // namespace vovp
#endif /* VOVP_PLASMA_MANAGER_H */
