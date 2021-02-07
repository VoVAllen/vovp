#ifndef VOVP_UTILS_H
#define VOVP_UTILS_H

#include <dmlc/logging.h>

#include <plasma/client.h>
#include <plasma/common.h>
#define VOVP_CHECK_ARROW(status)                                               \
  do {                                                                         \
    CHECK(status.ok()) << "Fail: " << status.ToString();                       \
  } while (0)

namespace vovp {
using plasma::ObjectID;

inline void check_arrow_status(const Status &status) {
  CHECK(status.ok()) << "Fail: " << status.ToString();
}

inline ObjectID ToObjectID(std::string object_id) {
  // CHECK(object_id.size()<20);
  ObjectID id;
  std::memset(id.mutable_data(), 0, plasma::kUniqueIDSize);
  std::memcpy(id.mutable_data(), object_id.data(), object_id.size());
  // obj.from_binary(object_id);
  return id;
}
} // namespace vovp

#endif /* VOVP_UTILS_H */
