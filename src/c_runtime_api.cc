/*!
 *  Copyright (c) 2016 by Contributors
 * \file c_runtime_api.cc
 * \brief Runtime API implementation
 */
#include <dmlc/thread_local.h>
#include <vovp/c_runtime_api.h>
// #include <vovp/runtime/c_backend_api.h>
// #include <vovp/runtime/packed_func.h>
// #include <vovp/runtime/module.h>
// #include <vovp/runtime/registry.h>
// #include <vovp/runtime/device_api.h>
// #include <vovp/runtime/tensordispatch.h>
#include <array>
#include <algorithm>
#include <string>
#include <cstdlib>
#include "dlpack/dlpack.h"
#include "vovp/device_api.h"
#include <dmlc/logging.h>
#include <dmlc/registry.h>
// #include "runtime_base.h"

namespace dmlc{
  DMLC_REGISTRY_ENABLE(vovp::runtime::DeviceAPI);
}

namespace vovp {
namespace runtime {

/*!
 * \brief The name of Device API factory.
 * \param type The device type.
 */
inline std::string DeviceName(int type) {
  switch (type) {
    case kDLCPU: return "cpu";
    case kDLGPU: return "gpu";
    case kDLOpenCL: return "opencl";
    case kDLSDAccel: return "sdaccel";
    case kDLAOCL: return "aocl";
    case kDLVulkan: return "vulkan";
    case kDLMetal: return "metal";
    case kDLVPI: return "vpi";
    case kDLROCM: return "rocm";
    case kOpenGL: return "opengl";
    case kExtDev: return "ext_dev";
    default: LOG(FATAL) << "unknown type =" << type; return "Unknown";
  }
}

class DeviceAPIManager {
 public:
  static const int kMaxDeviceAPI = 32;
  // Get API
  static DeviceAPI* Get(const VOVPContext& ctx) {
    return Get(ctx.device_type);
  }
  static DeviceAPI* Get(int dev_type, bool allow_missing = false) {
    return Global()->GetAPI(dev_type, allow_missing);
  }

 private:
  std::array<DeviceAPI*, kMaxDeviceAPI> api_;
  DeviceAPI* rpc_api_{nullptr};
  std::mutex mutex_;
  // constructor
  DeviceAPIManager() {
    std::fill(api_.begin(), api_.end(), nullptr);
  }
  // Global static variable.
  static DeviceAPIManager* Global() {
    static DeviceAPIManager inst;
    return &inst;
  }
  // Get or initialize API.
  DeviceAPI* GetAPI(int type, bool allow_missing) {
    if (type < kRPCSessMask) {
      if (api_[type] != nullptr) return api_[type];
      std::lock_guard<std::mutex> lock(mutex_);
      if (api_[type] != nullptr) return api_[type];
      api_[type] = GetAPI(DeviceName(type), allow_missing);
      return api_[type];
    } else {
      if (rpc_api_ != nullptr) return rpc_api_;
      std::lock_guard<std::mutex> lock(mutex_);
      if (rpc_api_ != nullptr) return rpc_api_;
      rpc_api_ = GetAPI("rpc", allow_missing);
      return rpc_api_;
    }
  }
  DeviceAPI* GetAPI(const std::string name, bool allow_missing) {
    std::string factory = "device_api." + name;
    auto f = dmlc::Registry<DeviceAPI>::Find(factory);
    // auto* f = Registry::Get(factory);
    if (f == nullptr) {
      CHECK(allow_missing)
          << "Device API " << name << " is not enabled. Please install the cuda version of vovp.";
      return nullptr;
    }
    // void* ptr = (*f)();
    return const_cast<DeviceAPI*>(f);
  }
};

DeviceAPI* DeviceAPI::Get(VOVPContext ctx, bool allow_missing) {
  return DeviceAPIManager::Get(
      static_cast<int>(ctx.device_type), allow_missing);
}

void* DeviceAPI::AllocWorkspace(VOVPContext ctx,
                                size_t size,
                                VOVPType type_hint) {
  return AllocDataSpace(ctx, size, kTempAllocaAlignment, type_hint);
}

void DeviceAPI::FreeWorkspace(VOVPContext ctx, void* ptr) {
  FreeDataSpace(ctx, ptr);
}

VOVPStreamHandle DeviceAPI::CreateStream(VOVPContext ctx) {
  LOG(FATAL) << "Device does not support stream api.";
  return 0;
}

void DeviceAPI::FreeStream(VOVPContext ctx, VOVPStreamHandle stream) {
  LOG(FATAL) << "Device does not support stream api.";
}

void DeviceAPI::SyncStreamFromTo(VOVPContext ctx,
                                 VOVPStreamHandle event_src,
                                 VOVPStreamHandle event_dst) {
  LOG(FATAL) << "Device does not support stream api.";
}
}  // namespace runtime
}  // namespace vovp

using namespace vovp::runtime;

struct VOVPRuntimeEntry {
  std::string ret_str;
  std::string last_error;
  VOVPByteArray ret_bytes;
};

typedef dmlc::ThreadLocalStore<VOVPRuntimeEntry> VOVPAPIRuntimeStore;

const char *VOVPGetLastError() {
  return VOVPAPIRuntimeStore::Get()->last_error.c_str();
}

void VOVPAPISetLastError(const char* msg) {
#ifndef _LIBCPP_SGX_CONFIG
  VOVPAPIRuntimeStore::Get()->last_error = msg;
#else
  sgx::OCallPackedFunc("__sgx_set_last_error__", msg);
#endif
}

// int VOVPModLoadFromFile(const char* file_name,
//                        const char* format,
//                        VOVPModuleHandle* out) {
//   API_BEGIN();
//   Module m = Module::LoadFromFile(file_name, format);
//   *out = new Module(m);
//   API_END();
// }

// int VOVPModImport(VOVPModuleHandle mod,
//                  VOVPModuleHandle dep) {
//   API_BEGIN();
//   static_cast<Module*>(mod)->Import(
//       *static_cast<Module*>(dep));
//   API_END();
// }

// int VOVPModGetFunction(VOVPModuleHandle mod,
//                       const char* func_name,
//                       int query_imports,
//                       VOVPFunctionHandle *func) {
//   API_BEGIN();
//   PackedFunc pf = static_cast<Module*>(mod)->GetFunction(
//       func_name, query_imports != 0);
//   if (pf != nullptr) {
//     *func = new PackedFunc(pf);
//   } else {
//     *func = nullptr;
//   }
//   API_END();
// }

// int VOVPModFree(VOVPModuleHandle mod) {
//   API_BEGIN();
//   delete static_cast<Module*>(mod);
//   API_END();
// }

// int VOVPBackendGetFuncFromEnv(void* mod_node,
//                              const char* func_name,
//                              VOVPFunctionHandle *func) {
//   API_BEGIN();
//   *func = (VOVPFunctionHandle)(
//       static_cast<ModuleNode*>(mod_node)->GetFuncFromEnv(func_name));
//   API_END();
// }

void* VOVPBackendAllocWorkspace(int device_type,
                               int device_id,
                               uint64_t size,
                               int dtype_code_hint,
                               int dtype_bits_hint) {
  VOVPContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id = device_id;

  VOVPType type_hint;
  type_hint.code = static_cast<decltype(type_hint.code)>(dtype_code_hint);
  type_hint.bits = static_cast<decltype(type_hint.bits)>(dtype_bits_hint);
  type_hint.lanes = 1;

  return DeviceAPIManager::Get(ctx)->AllocWorkspace(ctx,
                                                    static_cast<size_t>(size),
                                                    type_hint);
}

int VOVPBackendFreeWorkspace(int device_type,
                            int device_id,
                            void* ptr) {
  VOVPContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id = device_id;
  DeviceAPIManager::Get(ctx)->FreeWorkspace(ctx, ptr);
  return 0;
}

// int VOVPBackendRunOnce(void** handle,
//                       int (*f)(void*),
//                       void* cdata,
//                       int nbytes) {
//   if (*handle == nullptr) {
//     *handle = reinterpret_cast<void*>(1);
//     return (*f)(cdata);
//   }
//   return 0;
// }

// int VOVPFuncFree(VOVPFunctionHandle func) {
//   API_BEGIN();
//   delete static_cast<PackedFunc*>(func);
//   API_END();
// }

// int VOVPFuncCall(VOVPFunctionHandle func,
//                 VOVPValue* args,
//                 int* arg_type_codes,
//                 int num_args,
//                 VOVPValue* ret_val,
//                 int* ret_type_code) {
//   API_BEGIN();
//   VOVPRetValue rv;
//   (*static_cast<const PackedFunc*>(func)).CallPacked(
//       VOVPArgs(args, arg_type_codes, num_args), &rv);
//   // handle return string.
//   if (rv.type_code() == kStr ||
//      rv.type_code() == kVOVPType ||
//       rv.type_code() == kBytes) {
//     VOVPRuntimeEntry* e = VOVPAPIRuntimeStore::Get();
//     if (rv.type_code() != kVOVPType) {
//       e->ret_str = *rv.ptr<std::string>();
//     } else {
//       e->ret_str = rv.operator std::string();
//     }
//     if (rv.type_code() == kBytes) {
//       e->ret_bytes.data = e->ret_str.c_str();
//       e->ret_bytes.size = e->ret_str.length();
//       *ret_type_code = kBytes;
//       ret_val->v_handle = &(e->ret_bytes);
//     } else {
//       *ret_type_code = kStr;
//       ret_val->v_str = e->ret_str.c_str();
//     }
//   } else {
//     rv.MoveToCHost(ret_val, ret_type_code);
//   }
//   API_END();
// }

// int VOVPCFuncSetReturn(VOVPRetValueHandle ret,
//                       VOVPValue* value,
//                       int* type_code,
//                       int num_ret) {
//   API_BEGIN();
//   CHECK_EQ(num_ret, 1);
//   VOVPRetValue* rv = static_cast<VOVPRetValue*>(ret);
//   *rv = VOVPArgValue(value[0], type_code[0]);
//   API_END();
// }

// int VOVPFuncCreateFromCFunc(VOVPPackedCFunc func,
//                            void* resource_handle,
//                            VOVPPackedCFuncFinalizer fin,
//                            VOVPFunctionHandle *out) {
//   API_BEGIN();
//   if (fin == nullptr) {
//     *out = new PackedFunc(
//         [func, resource_handle](VOVPArgs args, VOVPRetValue* rv) {
//           int ret = func((VOVPValue*)args.values, (int*)args.type_codes, // NOLINT(*)
//                          args.num_args, rv, resource_handle);
//           if (ret != 0) {
//             std::string err = "VOVPCall CFunc Error:\n";
//             err += VOVPGetLastError();
//             throw dmlc::Error(err);
//           }
//         });
//   } else {
//     // wrap it in a shared_ptr, with fin as deleter.
//     // so fin will be called when the lambda went out of scope.
//     std::shared_ptr<void> rpack(resource_handle, fin);
//     *out = new PackedFunc(
//         [func, rpack](VOVPArgs args, VOVPRetValue* rv) {
//           int ret = func((VOVPValue*)args.values, (int*)args.type_codes, // NOLINT(*)
//                          args.num_args, rv, rpack.get());
//           if (ret != 0) {
//             std::string err = "VOVPCall CFunc Error:\n";
//             err += VOVPGetLastError();
//             throw dmlc::Error(err);
//           }
//       });
//   }
//   API_END();
// }

// int VOVPStreamCreate(int device_type, int device_id, VOVPStreamHandle* out) {
//   API_BEGIN();
//   VOVPContext ctx;
//   ctx.device_type = static_cast<DLDeviceType>(device_type);
//   ctx.device_id = device_id;
//   *out = DeviceAPIManager::Get(ctx)->CreateStream(ctx);
//   API_END();
// }

// int VOVPStreamFree(int device_type, int device_id, VOVPStreamHandle stream) {
//   API_BEGIN();
//   VOVPContext ctx;
//   ctx.device_type = static_cast<DLDeviceType>(device_type);
//   ctx.device_id = device_id;
//   DeviceAPIManager::Get(ctx)->FreeStream(ctx, stream);
//   API_END();
// }

// int VOVPSetStream(int device_type, int device_id, VOVPStreamHandle stream) {
//   API_BEGIN();
//   VOVPContext ctx;
//   ctx.device_type = static_cast<DLDeviceType>(device_type);
//   ctx.device_id = device_id;
//   DeviceAPIManager::Get(ctx)->SetStream(ctx, stream);
//   API_END();
// }

// int VOVPSynchronize(int device_type, int device_id, VOVPStreamHandle stream) {
//   API_BEGIN();
//   VOVPContext ctx;
//   ctx.device_type = static_cast<DLDeviceType>(device_type);
//   ctx.device_id = device_id;
//   DeviceAPIManager::Get(ctx)->StreamSync(ctx, stream);
//   API_END();
// }

// int VOVPStreamStreamSynchronize(int device_type,
//                                int device_id,
//                                VOVPStreamHandle src,
//                                VOVPStreamHandle dst) {
//   API_BEGIN();
//   VOVPContext ctx;
//   ctx.device_type = static_cast<DLDeviceType>(device_type);
//   ctx.device_id = device_id;
//   DeviceAPIManager::Get(ctx)->SyncStreamFromTo(ctx, src, dst);
//   API_END();
// }

// int VOVPCbArgToReturn(VOVPValue* value, int code) {
//   API_BEGIN();
//   vovp::runtime::VOVPRetValue rv;
//   rv = vovp::runtime::VOVPArgValue(*value, code);
//   int tcode;
//   rv.MoveToCHost(value, &tcode);
//   CHECK_EQ(tcode, code);
//   API_END();
// }

// void VOVPLoadTensorAdapter(const char *path) {
//   TensorDispatcher::Global()->Load(path);
// }

// // set device api
// VOVP_REGISTER_GLOBAL(vovp::runtime::symbol::vovp_set_device)
// .set_body([](VOVPArgs args, VOVPRetValue *ret) {
//     VOVPContext ctx;
//     ctx.device_type = static_cast<DLDeviceType>(args[0].operator int());
//     ctx.device_id = args[1];
//     DeviceAPIManager::Get(ctx)->SetDevice(ctx);
//   });

// // set device api
// VOVP_REGISTER_GLOBAL("_GetDeviceAttr")
// .set_body([](VOVPArgs args, VOVPRetValue *ret) {
//     VOVPContext ctx;
//     ctx.device_type = static_cast<DLDeviceType>(args[0].operator int());
//     ctx.device_id = args[1];

//     DeviceAttrKind kind = static_cast<DeviceAttrKind>(args[2].operator int());
//     if (kind == kExist) {
//       DeviceAPI* api = DeviceAPIManager::Get(ctx.device_type, true);
//       if (api != nullptr) {
//         api->GetAttr(ctx, kind, ret);
//       } else {
//         *ret = 0;
//       }
//     } else {
//       DeviceAPIManager::Get(ctx)->GetAttr(ctx, kind, ret);
//     }
//   });

