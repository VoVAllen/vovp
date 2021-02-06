/*!
 *  Copyright (c) 2017 by Contributors
 * \file vovp/runtime/packed_func.h
 * \brief Type-erased function used across VOVP API.
 */
#ifndef VOVP_RUNTIME_PACKED_FUNC_H_
#define VOVP_RUNTIME_PACKED_FUNC_H_

#include <dmlc/logging.h>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
namespace vovp {
namespace runtime {

// Forward declare ObjectRef and Object for extensions.
// This header works fine without depend on ObjectRef
// as long as it is not used.
class Object;
class ObjectRef;

// forward declarations
class VOVPArgs;
class VOVPArgValue;
class VOVPRetValue;
class VOVPArgsSetter;

/*!
 * \brief Packed function is a type-erased function.
 *  The arguments are passed by packed format.
 *
 *  This is an useful unified interface to call generated functions,
 *  It is the unified function function type of VOVP.
 *  It corresponds to VOVPFunctionHandle in C runtime API.
 */
class PackedFunc {
public:
  /*!
   * \brief The internal std::function
   * \param args The arguments to the function.
   * \param rv The return value.
   *
   * \code
   *   // Example code on how to implemented FType
   *   void MyPackedFunc(VOVPArgs args, VOVPRetValue* rv) {
   *     // automatically convert arguments to desired type.
   *     int a0 = args[0];
   *     float a1 = args[1];
   *     ...
   *     // automatically assign values to rv
   *     std::string my_return_value = "x";
   *     *rv = my_return_value;
   *   }
   * \endcode
   */
  using FType = std::function<void(VOVPArgs args, VOVPRetValue *rv)>;
  /*! \brief default constructor */
  PackedFunc() {}
  /*!
   * \brief constructing a packed function from a std::function.
   * \param body the internal container of packed function.
   */
  explicit PackedFunc(FType body) : body_(body) {}
  /*!
   * \brief Call packed function by directly passing in unpacked format.
   * \param args Arguments to be passed.
   * \tparam Args arguments to be passed.
   *
   * \code
   *   // Example code on how to call packed function
   *   void CallPacked(PackedFunc f) {
   *     // call like normal functions by pass in arguments
   *     // return value is automatically converted back
   *     int rvalue = f(1, 2.0);
   *   }
   * \endcode
   */
  template <typename... Args>
  inline VOVPRetValue operator()(Args &&...args) const;
  /*!
   * \brief Call the function in packed format.
   * \param args The arguments
   * \param rv The return value.
   */
  inline void CallPacked(VOVPArgs args, VOVPRetValue *rv) const;
  /*! \return the internal body function */
  inline FType body() const;
  /*! \return Whether the packed function is nullptr */
  bool operator==(std::nullptr_t null) const { return body_ == nullptr; }
  /*! \return Whether the packed function is not nullptr */
  bool operator!=(std::nullptr_t null) const { return body_ != nullptr; }

private:
  /*! \brief internal container of packed function */
  FType body_;
};

/*!
 * \brief Please refer to \ref TypedPackedFuncAnchor
 * "TypedPackedFunc<R(Args..)>"
 */
template <typename FType> class TypedPackedFunc;

} // namespace runtime
} // namespace vovp

#endif  // DGL_RUNTIME_PACKED_FUNC_H_