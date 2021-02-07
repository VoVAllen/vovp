#include <pybind11/pybind11.h>
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
} // namespace vovp

PYBIND11_MODULE(_vovp, m) {

  using namespace vovp;
  py::class_<vovp::VovpPlasmaManager>(m, "VovpPlasmaClient")
      .def(py::init<const std::string &>())
      .def("put_tensor",
           [](vovp::VovpPlasmaManager &manager, const py::capsule &pycapsule,
              std::string object_id) {
             auto *dlm_ptr =
                 reinterpret_cast<DLManagedTensor *>(pycapsule.get_pointer());
             DLManagedTensor *new_dlm_ptr =
                 manager.put_dlpack_tensor(dlm_ptr, object_id);

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
             auto *dlm_ptr = manager.get_dlpack_tensor(object_id);
             py::capsule new_capsule(dlm_ptr, "dltensor",
                                     &DlpackCapsuleDestructor);
             return new_capsule;
           })
      .def("list", [](vovp::VovpPlasmaManager &manager) {
        ObjectTable table;
        VOVP_CHECK_ARROW(manager.client->List(&table));
        LOG(INFO) << "Table size: " << table.size();
        for (const auto &kv : table) {
          LOG(INFO) << "DL Ref: " << kv.first.hex() << ": "
                    << kv.second->ref_count;
        }
        // auto status = owner->plasma_client->Delete(owner->object_id);
        // LOG(INFO) << "Delete: " << owner->object_id.hex();
        // check_arrow_status(status);
        //   LOG(INFO) << "buf Delete: " << object_id.hex();
        //   VOVP_CHECK_ARROW(client->Delete(object_id));
        // };
        // return new_capsule;
      });

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
