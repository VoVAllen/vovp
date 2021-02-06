/*!
 *  Copyright (c) 2016 by Contributors
 * \file vovp/runtime/c_runtime_api.h
 * \brief VOVP runtime library.
 *
 * This runtime is adapted from TVM project (commit: 2ce5277)
 */
#ifndef VOVP_RUNTIME_C_RUNTIME_API_H_
#define VOVP_RUNTIME_C_RUNTIME_API_H_

// Macros to do weak linking
#ifdef _MSC_VER
#define VOVP_WEAK __declspec(selectany)
#else
#define VOVP_WEAK __attribute__((weak))
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define VOVP_DLL EMSCRIPTEN_KEEPALIVE
#endif

#ifndef VOVP_DLL
#ifdef _WIN32
#ifdef VOVP_EXPORTS
#define VOVP_DLL __declspec(dllexport)
#else
#define VOVP_DLL __declspec(dllimport)
#endif
#else
#define VOVP_DLL
#endif
#endif

// VOVP version
#define VOVP_VERSION "0.6"


// VOVP Runtime is DLPack compatible.
#include <dlpack/dlpack.h>

#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>
#include <stddef.h>

/*! \brief type of array index. */
typedef int64_t vovp_index_t;

/*! \brief Extension device types in VOVP */
typedef enum {
  kDLAOCL = 5,
  kDLSDAccel = 6,
  kOpenGL = 11,
  // Extension DRAM type, used for quickly test extension device
  // The device api can differ depending on the xpu driver registered.
  kExtDev = 12,
  // AddExtraVOVPType which is not in DLPack here
} VOVPDeviceExtType;

/*!
 * \brief The type code in VOVPType
 * \note VOVPType is used in two places.
 */
typedef enum {
  // The type code of other types are compatible with DLPack.
  // The next few fields are extension types
  // that is used by VOVP API calls.
  kHandle = 3U,
  kNull = 4U,
  kVOVPType = 5U,
  kVOVPContext = 6U,
  kArrayHandle = 7U,
  kObjectHandle = 8U,
  kModuleHandle = 9U,
  kFuncHandle = 10U,
  kStr = 11U,
  kBytes = 12U,
  kNDArrayContainer = 13U,
  // Extension codes for other frameworks to integrate VOVP PackedFunc.
  // To make sure each framework's id do not conflict, use first and
  // last sections to mark ranges.
  // Open an issue at the repo if you need a section of code.
  kExtBegin = 15U,
  kNNVMFirst = 16U,
  kNNVMLast = 20U,
  // The following section of code is used for non-reserved types.
  kExtReserveEnd = 64U,
  kExtEnd = 128U
} VOVPTypeCode;

/*!
 * \brief The data type used in VOVP Runtime.
 *
 *  Examples
 *   - float: type_code = 2, bits = 32, lanes=1
 *   - float4(vectorized 4 float): type_code = 2, bits = 32, lanes=4
 *   - int8: type_code = 0, bits = 8, lanes=1
 *
 * \note Arguments VOVP API function always takes bits=64 and lanes=1
 */
typedef DLDataType VOVPType;

/*!
 * \brief The Device information, abstract away common device types.
 */
typedef DLContext VOVPContext;

/*!
 * \brief The tensor array stucture to VOVP API.
 */
typedef DLTensor VOVPArray;

/*! \brief the array handle */
typedef VOVPArray* VOVPArrayHandle;

/*!
 * \brief Union type of values
 *  being passed through API and function calls.
 */
typedef union {
  int64_t v_int64;
  double v_float64;
  void* v_handle;
  const char* v_str;
  VOVPType v_type;
  VOVPContext v_ctx;
} VOVPValue;

/*!
 * \brief Byte array type used to pass in byte array
 *  When kBytes is used as data type.
 */
typedef struct {
  const char* data;
  size_t size;
} VOVPByteArray;

/*! \brief Handle to VOVP runtime modules. */
typedef void* VOVPModuleHandle;
/*! \brief Handle to packed function handle. */
typedef void* VOVPFunctionHandle;
/*! \brief Handle to hold return value. */
typedef void* VOVPRetValueHandle;
/*!
 * \brief The stream that is specific to device
 * can be NULL, which indicates the default one.
 */
typedef void* VOVPStreamHandle;

/*!
 * \brief Used for implementing C API function.
 *  Set last error message before return.
 * \param msg The error message to be set.
 */
VOVP_DLL void VOVPAPISetLastError(const char* msg);

/*!
 * \brief return str message of the last error
 *  all function in this file will return 0 when success
 *  and -1 when an error occured,
 *  VOVPGetLastError can be called to retrieve the error
 *
 *  this function is threadsafe and can be called by different thread
 *  \return error info
 */
VOVP_DLL const char *VOVPGetLastError(void);
/*!
 * \brief Load module from file.
 * \param file_name The file name to load the module from.
 * \param format The format of the module.
 * \param out The result module
 *
 * \return 0 when success, -1 when failure happens
 * \note The resulting module do not contain import relation.
 *  It can be reconstructed by VOVPModImport.
 */
VOVP_DLL int VOVPModLoadFromFile(const char* file_name,
                               const char* format,
                               VOVPModuleHandle* out);

/*!
 * \brief Add dep to mod's dependency.
 *  This allows functions in this module to use modules.
 *
 * \param mod The module handle.
 * \param dep The dependent module to be imported.
 * \return 0 when success, -1 when failure happens
 */
VOVP_DLL int VOVPModImport(VOVPModuleHandle mod,
                         VOVPModuleHandle dep);

/*!
 * \brief Get function from the module.
 * \param mod The module handle.
 * \param func_name The name of the function.
 * \param query_imports Whether to query imported modules
 * \param out The result function, can be NULL if it is not available.
 * \return 0 when no error is thrown, -1 when failure happens
 */
VOVP_DLL int VOVPModGetFunction(VOVPModuleHandle mod,
                              const char* func_name,
                              int query_imports,
                              VOVPFunctionHandle *out);

/*!
 * \brief Free front-end extension type resource.
 * \param handle The extension handle.
 * \param type_code The type of of the extension type.
 * \return 0 when success, -1 when failure happens
 */
VOVP_DLL int VOVPExtTypeFree(void* handle, int type_code);

/*!
 * \brief Free the Module
 * \param mod The module to be freed.
 *
 * \note This may not free up the module's resources.
 *  If there is active VOVPFunctionHandle uses the module
 *  Or if this module is imported by another active module.
 *
 *  The all functions remains valid until VOVPFuncFree is called.
 * \return 0 when success, -1 when failure happens
 */
VOVP_DLL int VOVPModFree(VOVPModuleHandle mod);

/*!
 * \brief Free the function when it is no longer needed.
 * \param func The function handle
 * \return 0 when success, -1 when failure happens
 */
VOVP_DLL int VOVPFuncFree(VOVPFunctionHandle func);

/*!
 * \brief Call a Packed VOVP Function.
 *
 * \param func node handle of the function.
 * \param arg_values The arguments
 * \param type_codes The type codes of the arguments
 * \param num_args Number of arguments.
 *
 * \param ret_val The return value.
 * \param ret_type_code the type code of return value.
 *
 * \return 0 when success, -1 when failure happens
 * \note VOVP calls always exchanges with type bits=64, lanes=1
 *
 * \note API calls always exchanges with type bits=64, lanes=1
 *   If API call returns container handles (e.g. FunctionHandle)
 *   these handles should be managed by the front-end.
 *   The front-end need to call free function (e.g. VOVPFuncFree)
 *   to free these handles.
 */
VOVP_DLL int VOVPFuncCall(VOVPFunctionHandle func,
                        VOVPValue* arg_values,
                        int* type_codes,
                        int num_args,
                        VOVPValue* ret_val,
                        int* ret_type_code);

/*!
 * \brief Set the return value of VOVPPackedCFunc.
 *
 *  This function is called by VOVPPackedCFunc to set the return value.
 *  When this function is not called, the function returns null by default.
 *
 * \param ret The return value handle, pass by ret in VOVPPackedCFunc
 * \param value The value to be returned.
 * \param type_code The type of the value to be returned.
 * \param num_ret Number of return values, for now only 1 is supported.
 */
VOVP_DLL int VOVPCFuncSetReturn(VOVPRetValueHandle ret,
                              VOVPValue* value,
                              int* type_code,
                              int num_ret);

/*!
 * \brief Inplace translate callback argument value to return value.
 *  This is only needed for non-POD arguments.
 *
 * \param value The value to be translated.
 * \param code The type code to be translated.
 * \note This function will do a shallow copy when necessary.
 *
 * \return 0 when success, -1 when failure happens.
 */
VOVP_DLL int VOVPCbArgToReturn(VOVPValue* value, int code);

/*!
 * \brief C type of packed function.
 *
 * \param args The arguments
 * \param type_codes The type codes of the arguments
 * \param num_args Number of arguments.
 * \param ret The return value handle.
 * \param resource_handle The handle additional resouce handle from fron-end.
 * \return 0 if success, -1 if failure happens, set error via VOVPAPISetLastError.
 * \sa VOVPCFuncSetReturn
 */
typedef int (*VOVPPackedCFunc)(
    VOVPValue* args,
    int* type_codes,
    int num_args,
    VOVPRetValueHandle ret,
    void* resource_handle);

/*!
 * \brief C callback to free the resource handle in C packed function.
 * \param resource_handle The handle additional resouce handle from fron-end.
 */
typedef void (*VOVPPackedCFuncFinalizer)(void* resource_handle);

/*!
 * \brief Signature for extension function declarer.
 *
 *  VOVP call this function to get the extension functions
 *  The declarer will call register_func to register function and their name.
 *
 * \param register_func_handle The register function
 * \return 0 if success, -1 if failure happens
 */
typedef int (*VOVPExtensionFuncDeclarer)(VOVPFunctionHandle register_func_handle);

/*!
 * \brief Wrap a VOVPPackedCFunc to become a FunctionHandle.
 *
 * The resource_handle will be managed by VOVP API, until the function is no longer used.
 *
 * \param func The packed C function.
 * \param resource_handle The resource handle from front-end, can be NULL.
 * \param fin The finalizer on resource handle when the FunctionHandle get freed, can be NULL
 * \param out the result function handle.
 * \return 0 when success, -1 when failure happens
 */
VOVP_DLL int VOVPFuncCreateFromCFunc(VOVPPackedCFunc func,
                                   void* resource_handle,
                                   VOVPPackedCFuncFinalizer fin,
                                   VOVPFunctionHandle *out);

/*!
 * \brief Register the function to runtime's global table.
 *
 * The registered function then can be pulled by the backend by the name.
 *
 * \param name The name of the function.
 * \param f The function to be registered.
 * \param override Whether allow override already registered function.
 */
VOVP_DLL int VOVPFuncRegisterGlobal(
    const char* name, VOVPFunctionHandle f, int override);

/*!
 * \brief Get a global function.
 *
 * \param name The name of the function.
 * \param out the result function pointer, NULL if it does not exist.
 *
 * \note The function handle of global function is managed by VOVP runtime,
 *  So VOVPFuncFree is should not be called when it get deleted.
 */
VOVP_DLL int VOVPFuncGetGlobal(const char* name, VOVPFunctionHandle* out);

/*!
 * \brief List all the globally registered function name
 * \param out_size The number of functions
 * \param out_array The array of function names.
 * \return 0 when success, -1 when failure happens
 */
VOVP_DLL int VOVPFuncListGlobalNames(int* out_size,
                                   const char*** out_array);

// Array related apis for quick proptyping
/*!
 * \brief Allocate a nd-array's memory,
 *  including space of shape, of given spec.
 *
 * \param shape The shape of the array, the data content will be copied to out
 * \param ndim The number of dimension of the array.
 * \param dtype_code The type code of the dtype
 * \param dtype_bits The number of bits of dtype
 * \param dtype_lanes The number of lanes in the dtype.
 * \param device_type The device type of context
 * \param device_id The device id of context.
 * \param out The output handle.
 * \return 0 when success, -1 when failure happens
 */
VOVP_DLL int VOVPArrayAlloc(const vovp_index_t* shape,
                          int ndim,
                          int dtype_code,
                          int dtype_bits,
                          int dtype_lanes,
                          int device_type,
                          int device_id,
                          VOVPArrayHandle* out);

/*!
 * \brief Allocate a nd-array's with shared memory,
 *  including space of shape, of given spec.
 *
 * \param the name of the shared memory
 * \param shape The shape of the array, the data content will be copied to out
 * \param ndim The number of dimension of the array.
 * \param dtype_code The type code of the dtype
 * \param dtype_bits The number of bits of dtype
 * \param dtype_lanes The number of lanes in the dtype.
 * \param is_create whether the shared memory is created
 * \param out The output handle.
 * \return 0 when success, -1 when failure happens
 */
int VOVPArrayAllocSharedMem(const char *mem_name,
                           const vovp_index_t *shape,
                           int ndim,
                           int dtype_code,
                           int dtype_bits,
                           int dtype_lanes,
                           bool is_create,
                           VOVPArrayHandle* out);

/*!
 * \brief Free the VOVP Array.
 * \param handle The array handle to be freed.
 * \return 0 when success, -1 when failure happens
 */
VOVP_DLL int VOVPArrayFree(VOVPArrayHandle handle);

/*!
 * \brief Copy array data from CPU byte array.
 * \param handle The array handle.
 * \param data the data pointer
 * \param nbytes The number of bytes to copy.
 * \return 0 when success, -1 when failure happens
 */
VOVP_DLL int VOVPArrayCopyFromBytes(VOVPArrayHandle handle,
                                  void* data,
                                  size_t nbytes);

/*!
 * \brief Copy array data to CPU byte array.
 * \param handle The array handle.
 * \param data the data pointer
 * \param nbytes The number of bytes to copy.
 * \return 0 when success, -1 when failure happens
 */
VOVP_DLL int VOVPArrayCopyToBytes(VOVPArrayHandle handle,
                                void* data,
                                size_t nbytes);

/*!
 * \brief Copy the array, both from and to must be valid during the copy.
 * \param from The array to be copied from.
 * \param to The target space.
 * \param stream The stream where the copy happens, can be NULL.
 * \return 0 when success, -1 when failure happens
 */
VOVP_DLL int VOVPArrayCopyFromTo(VOVPArrayHandle from,
                               VOVPArrayHandle to,
                               VOVPStreamHandle stream);

/*!
 * \brief Produce an array from the DLManagedTensor that shares data memory
 * with the DLManagedTensor.
 * \param from The source DLManagedTensor.
 * \param out The output array handle.
 * \return 0 when success, -1 when failure happens
 */
VOVP_DLL int VOVPArrayFromDLPack(DLManagedTensor* from,
                               VOVPArrayHandle* out);

/*!
 * \brief Produce a DLMangedTensor from the array that shares data memory with
 * the array.
 * \param from The source array.
 * \param out The DLManagedTensor handle.
 * \return 0 when success, -1 when failure happens
 */
VOVP_DLL int VOVPArrayToDLPack(VOVPArrayHandle from, DLManagedTensor** out,
                             int alignment = 0);

/*!
 * \brief Delete (free) a DLManagedTensor's data.
 * \param dltensor Pointer to the DLManagedTensor.
 */
VOVP_DLL void VOVPDLManagedTensorCallDeleter(DLManagedTensor* dltensor);

/*!
 * \brief Create a new runtime stream.
 *
 * \param device_type The device type of context
 * \param device_id The device id of context
 * \param out The new stream handle
 * \return 0 when success, -1 when failure happens
 */
VOVP_DLL int VOVPStreamCreate(int device_type, int device_id, VOVPStreamHandle* out);

/*!
 * \brief Free a created stream handle.
 *
 * \param device_type The device type of context
 * \param device_id The device id of context
 * \param stream The stream to be freed
 * \return 0 when success, -1 when failure happens
 */
VOVP_DLL int VOVPStreamFree(int device_type, int device_id, VOVPStreamHandle stream);

/*!
 * \brief Set the runtime stream of current thread to be stream.
 *  The subsequent calls to the same device_type
 *  will use the setted stream handle.
 *  The specific type of stream is runtime device dependent.
 *
 * \param device_type The device type of context
 * \param device_id The device id of context.
 * \param handle The stream handle.
 * \return 0 when success, -1 when failure happens
 */
VOVP_DLL int VOVPSetStream(int device_type, int device_id, VOVPStreamHandle handle);

/*!
 * \brief Wait until all computations on stream completes.
 *
 * \param device_type The device type of context
 * \param device_id The device id of context.
 * \param stream The stream to be synchronized.
 * \return 0 when success, -1 when failure happens
 */
VOVP_DLL int VOVPSynchronize(int device_type, int device_id, VOVPStreamHandle stream);

/*!
 * \brief Synchronize two streams of execution.
 *
 * \param device_type The device type of context
 * \param device_id The device id of context
 * \param src The source stream to synchronize.
 * \param dst The destination stream to synchronize.
 * \return 0 when success, -1 when failure happens
 */
VOVP_DLL int VOVPStreamStreamSynchronize(int device_type,
                                       int device_id,
                                       VOVPStreamHandle src,
                                       VOVPStreamHandle dst);

/*!
 * \brief Load tensor adapter.
 */
VOVP_DLL void VOVPLoadTensorAdapter(const char *path);

/*!
 * \brief Bug report macro.
 *
 * This serves as a sanity check on system side to make sure the code is correct by
 * checking whether a condition always holds for complex reasons.  Failing the
 * condition signifies a system bug instead of users giving invalid inputs or using
 * the functionality incorrectly.
 *
 * Hints the user to file a bug report if the condition fails.
 */
#define BUG_ON(cond) \
  CHECK(cond) << "A bug has been occurred.  " \
                 "Please file a bug report at https://github.com/dmlc/vovp/issues.  " \
                 "Message: "

#ifdef __cplusplus
}  // VOVP_EXTERN_C
#endif
#endif  // VOVP_RUNTIME_C_RUNTIME_API_H_
