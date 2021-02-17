#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <vovp/plasma_manager.h>
#include <vovp/utils.h>
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
namespace py = pybind11;

namespace vovp {
void DlpackCapsuleDestructor(PyObject *capsule) {
  if (PyCapsule_IsValid(capsule, "dltensor")) {
    auto *dlm_rptr = reinterpret_cast<DLManagedTensor *>(
        PyCapsule_GetPointer(capsule, "dltensor"));
    if (dlm_rptr) {
      if (dlm_rptr->deleter != nullptr) {
        dlm_rptr->deleter(dlm_rptr);
      }
      PyCapsule_SetDestructor(capsule, nullptr);
    }
  }
}

ObjectID BytesToObjectID(std::string &object_id) {
  return ToObjectID(object_id);
}
} // namespace vovp

PYBIND11_MODULE(_vovp, m) {

  using namespace vovp;
  py::class_<vovp::VovpPlasmaManager>(m, "VovpPlasmaClient")
      .def(py::init<const std::string &>())
      .def("put_tensor",
           [](vovp::VovpPlasmaManager &manager, const py::capsule &pycapsule,
              py::bytes object_id, bool release_when_destruct,
              bool try_delete_when_destruct, bool try_delete_before_create) {
             auto *dlm_ptr =
                 reinterpret_cast<DLManagedTensor *>(pycapsule.get_pointer());
             ObjectID plasma_object_id = ToObjectID(object_id);
             DLManagedTensor *new_dlm_ptr = manager.PutDlpackTensor(
                 dlm_ptr, plasma_object_id, release_when_destruct,
                 try_delete_when_destruct, try_delete_before_create);

             PyCapsule_SetName(pycapsule.ptr(), "used_dltensor");
             PyCapsule_SetDestructor(pycapsule.ptr(), nullptr);
             if (dlm_ptr->deleter != nullptr) {
               dlm_ptr->deleter(dlm_ptr);
             }
             py::capsule new_capsule(new_dlm_ptr, "dltensor",
                                     &DlpackCapsuleDestructor);
             return new_capsule;
           })
      .def("get_tensor",
           [](vovp::VovpPlasmaManager &manager, std::string object_id) {
             
             ObjectID plasma_object_id = ToObjectID(object_id);
             auto *dlm_ptr = manager.GetDlpackTensor(plasma_object_id);
             
             py::capsule new_capsule(dlm_ptr, "dltensor",
                                     &DlpackCapsuleDestructor);
             return new_capsule;
           })
      .def("list",
           [](vovp::VovpPlasmaManager &manager) {
             ObjectTable table;
             VOVP_CHECK_ARROW(manager.client->List(&table));
             LOG(INFO) << "Table size: " << table.size();
             for (const auto &kv : table) {
               LOG(INFO) << "DL Ref: " << kv.first.hex() << ": "
                         << kv.second->ref_count;
             }
           })
      .def("release",
           [](vovp::VovpPlasmaManager &manager, std::string object_id) {             
             ObjectID plasma_object_id = ToObjectID(object_id);
             manager.Release(plasma_object_id);
           });

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
