// Copyright 2024-2025
//
// Licensed under the Apache License, Version 2.0 (the "License")
// See the License at http://www.apache.org/licenses/LICENSE-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <sstream>
#include "kdtree.h"

namespace py = pybind11;
using namespace kdtree;

// Helper to convert Python input to Point
template<typename PointType>
PointType to_point(py::handle obj) {
    using T = typename PointType::value_type;

    if (py::isinstance<PointType>(obj)) {
        return obj.cast<PointType>();
    }

    if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
        auto seq = obj.cast<py::sequence>();
        if (seq.size() != 2) {
            throw std::runtime_error("Point requires exactly 2 coordinates");
        }
        return PointType(seq[0].cast<T>(), seq[1].cast<T>());
    }

    if (py::isinstance<py::buffer>(obj)) {
        py::buffer_info info = obj.cast<py::buffer>().request();
        if (info.ndim != 1 || info.shape[0] != 2) {
            throw std::runtime_error("Point buffer must be 1D with size 2");
        }
        if (info.format == py::format_descriptor<double>::format()) {
            double* ptr = static_cast<double*>(info.ptr);
            return PointType(static_cast<T>(ptr[0]), static_cast<T>(ptr[1]));
        } else if (info.format == py::format_descriptor<float>::format()) {
            float* ptr = static_cast<float*>(info.ptr);
            return PointType(static_cast<T>(ptr[0]), static_cast<T>(ptr[1]));
        }
        // Fallback for other types or complex buffer layouts
        auto seq = obj.cast<py::sequence>();
        return PointType(seq[0].cast<T>(), seq[1].cast<T>());
    }

    throw std::runtime_error("Cannot convert to Point - expected Point, tuple, list, or 1D buffer of size 2");
}

template<typename PointType, typename ValueType>
void bind_kdtree(py::module_& m, const std::string& name) {
    using Tree = KDTree<PointType, ValueType>;
    using Entry = typename Tree::Entry;
    using T = typename PointType::value_type;

    auto tree_class = py::class_<Tree>(m, name.c_str())
        .def(py::init<>())
        .def(py::init<const std::vector<Entry>&>())
        .def(py::init([](py::buffer coords, py::handle values) {
            py::buffer_info coords_info = coords.request();
            if (coords_info.ndim != 2 || coords_info.shape[1] != 2) {
                throw std::runtime_error("Coordinates must be an Nx2 array");
            }
            size_t n = coords_info.shape[0];
            const T* coords_ptr = static_cast<const T*>(coords_info.ptr);

            std::vector<Entry> entries;
            entries.reserve(n);

            if (py::isinstance<py::buffer>(values)) {
                // Optimized path for numeric values (like int64 IDs)
                py::buffer_info val_info = values.cast<py::buffer>().request();
                if (val_info.size != (ssize_t)n) {
                    throw std::runtime_error("Values length must match coordinates");
                }
                const ValueType* val_ptr = static_cast<const ValueType*>(val_info.ptr);
                for (size_t i = 0; i < n; ++i) {
                    entries.push_back({PointType(coords_ptr[i*2], coords_ptr[i*2+1]), val_ptr[i]});
                }
            } else {
                // Generic path for list of Python objects
                auto val_seq = values.cast<py::sequence>();
                if (val_seq.size() != n) {
                    throw std::runtime_error("Values length must match coordinates");
                }
                for (size_t i = 0; i < n; ++i) {
                    entries.push_back({PointType(coords_ptr[i*2], coords_ptr[i*2+1]), val_seq[i].cast<ValueType>()});
                }
            }
            return new Tree(entries);
        }))
        .def("empty", &Tree::empty)
        .def("size", &Tree::size)
        .def("clear", &Tree::clear)

        // insert() overloads - key (point) first, value second
        .def("insert", static_cast<bool (Tree::*)(Entry)>(&Tree::insert), py::arg("entry"))
        .def("insert", [](Tree& self, const PointType& p, ValueType val) {
            return self.insert(p, val);
        }, py::arg("point"), py::arg("value"))
        .def("insert", [](Tree& self, py::handle point_like, ValueType val) {
            return self.insert(to_point<PointType>(point_like), val);
        }, py::arg("point"), py::arg("value"))
        .def("insert", [](Tree& self, T x, T y, ValueType val) {
            return self.insert(PointType(x, y), val);
        }, py::arg("x"), py::arg("y"), py::arg("value"))

        // remove() overloads
        .def("remove", static_cast<bool (Tree::*)(PointType)>(&Tree::remove), py::arg("point"))
        .def("remove", [](Tree& self, py::handle point_like) {
            return self.remove(to_point<PointType>(point_like));
        }, py::arg("point"))
        .def("remove", [](Tree& self, T x, T y) {
            return self.remove(PointType(x, y));
        }, py::arg("x"), py::arg("y"))

        // exists() overloads
        .def("exists", static_cast<bool (Tree::*)(PointType) const>(&Tree::exists), py::arg("point"))
        .def("exists", [](const Tree& self, py::handle point_like) {
            return self.exists(to_point<PointType>(point_like));
        }, py::arg("point"))
        .def("exists", [](const Tree& self, T x, T y) {
            return self.exists(PointType(x, y));
        }, py::arg("x"), py::arg("y"))

        // find() overloads
        .def("find", static_cast<std::optional<Entry> (Tree::*)(PointType) const>(&Tree::find), py::arg("point"))
        .def("find", [](const Tree& self, py::handle point_like) {
            return self.find(to_point<PointType>(point_like));
        }, py::arg("point"))
        .def("find", [](const Tree& self, T x, T y) {
            return self.find(PointType(x, y));
        }, py::arg("x"), py::arg("y"))

        // find_closest supporting (point), (x,y), (point, max_dist), (x,y, max_dist) + metric/kwargs
        .def("find_closest", [](const Tree& self, py::args args, py::kwargs kwargs) -> std::optional<Entry> {
            PointType p;
            double max_dist = -1.0;
            py::handle metric_handle = py::none();

            size_t n_args = args.size();
            if (n_args == 0) throw py::value_error("Point or x,y coordinates required");

            size_t next_idx = 0;
            try {
                p = to_point<PointType>(args[0]);
                next_idx = 1;
            } catch (...) {
                if (n_args < 2) throw;
                p = PointType(args[0].cast<T>(), args[1].cast<T>());
                next_idx = 2;
            }

            for (size_t i = next_idx; i < n_args; ++i) {
                if (py::isinstance<py::float_>(args[i]) || py::isinstance<py::int_>(args[i])) {
                    max_dist = args[i].cast<double>();
                } else {
                    metric_handle = args[i];
                }
            }

            if (kwargs.contains("max_dist")) max_dist = kwargs["max_dist"].cast<double>();
            if (kwargs.contains("metric")) metric_handle = kwargs["metric"];

            if (max_dist >= 0 && metric_handle.is_none())
                throw py::value_error("Metric must be specified when max_dist is provided");

            if (metric_handle.is_none() || py::isinstance<L2sq>(metric_handle))
                return self.template find_closest<L2sq>(p, metric_handle.is_none() ? L2sq{} : metric_handle.cast<L2sq>(), max_dist);
            if (py::isinstance<L1>(metric_handle))
                return self.template find_closest<L1>(p, metric_handle.cast<L1>(), max_dist);
            if (py::isinstance<L2>(metric_handle))
                return self.template find_closest<L2>(p, metric_handle.cast<L2>(), max_dist);
            if (py::isinstance<Linf>(metric_handle))
                return self.template find_closest<Linf>(p, metric_handle.cast<Linf>(), max_dist);
            if (py::isinstance<Toroidal<L1, double>>(metric_handle))
                return self.template find_closest<Toroidal<L1, double>>(p, metric_handle.cast<Toroidal<L1, double>>(), max_dist);
            if (py::isinstance<Toroidal<L2, double>>(metric_handle))
                return self.template find_closest<Toroidal<L2, double>>(p, metric_handle.cast<Toroidal<L2, double>>(), max_dist);
            if (py::isinstance<Toroidal<L2sq, double>>(metric_handle))
                return self.template find_closest<Toroidal<L2sq, double>>(p, metric_handle.cast<Toroidal<L2sq, double>>(), max_dist);
            if (py::isinstance<Toroidal<Linf, double>>(metric_handle))
                return self.template find_closest<Toroidal<Linf, double>>(p, metric_handle.cast<Toroidal<Linf, double>>(), max_dist);
            if (py::isinstance<GreatCircle>(metric_handle))
                return self.template find_closest<GreatCircle>(p, metric_handle.cast<GreatCircle>(), max_dist);

            throw py::type_error("Unsupported metric type");
        })

        // find_closest_k supporting (point, k), (x,y, k) + max_dist/metric/kwargs
        .def("find_closest_k", [](const Tree& self, py::args args, py::kwargs kwargs) -> std::vector<Entry> {
            PointType p;
            size_t k = 1;
            double max_dist = -1.0;
            py::handle metric_handle = py::none();

            size_t n_args = args.size();
            if (n_args == 0) throw py::value_error("Point or x,y coordinates required");

            size_t next_idx = 0;
            try {
                p = to_point<PointType>(args[0]);
                next_idx = 1;
            } catch (...) {
                if (n_args < 2) throw;
                p = PointType(args[0].cast<T>(), args[1].cast<T>());
                next_idx = 2;
            }

            if (next_idx < n_args) {
                k = args[next_idx].cast<size_t>();
                next_idx++;
            }

            for (size_t i = next_idx; i < n_args; ++i) {
                if (py::isinstance<py::float_>(args[i]) || py::isinstance<py::int_>(args[i])) {
                    max_dist = args[i].cast<double>();
                } else {
                    metric_handle = args[i];
                }
            }

            if (kwargs.contains("k")) k = kwargs["k"].cast<size_t>();
            if (kwargs.contains("max_dist")) max_dist = kwargs["max_dist"].cast<double>();
            if (kwargs.contains("metric")) metric_handle = kwargs["metric"];

            if (max_dist >= 0 && metric_handle.is_none())
                throw py::value_error("Metric must be specified when max_dist is provided");

            if (metric_handle.is_none() || py::isinstance<L2sq>(metric_handle))
                return self.template find_closest_k<L2sq>(p, k, metric_handle.is_none() ? L2sq{} : metric_handle.cast<L2sq>(), max_dist);
            if (py::isinstance<L1>(metric_handle))
                return self.template find_closest_k<L1>(p, k, metric_handle.cast<L1>(), max_dist);
            if (py::isinstance<L2>(metric_handle))
                return self.template find_closest_k<L2>(p, k, metric_handle.cast<L2>(), max_dist);
            if (py::isinstance<Linf>(metric_handle))
                return self.template find_closest_k<Linf>(p, k, metric_handle.cast<Linf>(), max_dist);
            if (py::isinstance<Toroidal<L1, double>>(metric_handle))
                return self.template find_closest_k<Toroidal<L1, double>>(p, k, metric_handle.cast<Toroidal<L1, double>>(), max_dist);
            if (py::isinstance<Toroidal<L2, double>>(metric_handle))
                return self.template find_closest_k<Toroidal<L2, double>>(p, k, metric_handle.cast<Toroidal<L2, double>>(), max_dist);
            if (py::isinstance<Toroidal<L2sq, double>>(metric_handle))
                return self.template find_closest_k<Toroidal<L2sq, double>>(p, k, metric_handle.cast<Toroidal<L2sq, double>>(), max_dist);
            if (py::isinstance<Toroidal<Linf, double>>(metric_handle))
                return self.template find_closest_k<Toroidal<Linf, double>>(p, k, metric_handle.cast<Toroidal<Linf, double>>(), max_dist);
            if (py::isinstance<GreatCircle>(metric_handle))
                return self.template find_closest_k<GreatCircle>(p, k, metric_handle.cast<GreatCircle>(), max_dist);

            throw py::type_error("Unsupported metric type");
        })

        // find_all_within supporting (point, radius), (x,y, radius) + metric/kwargs
        .def("find_all_within", [](const Tree& self, py::args args, py::kwargs kwargs) -> std::vector<Entry> {
            PointType p;
            double radius = 0;
            py::handle metric_handle = py::none();

            size_t n_args = args.size();
            if (n_args == 0) throw py::value_error("Point or x,y coordinates required");

            size_t next_idx = 0;
            try {
                p = to_point<PointType>(args[0]);
                next_idx = 1;
            } catch (...) {
                if (n_args < 2) throw;
                p = PointType(args[0].cast<T>(), args[1].cast<T>());
                next_idx = 2;
            }

            if (next_idx < n_args) {
                radius = args[next_idx].cast<double>();
                next_idx++;
            }

            if (next_idx < n_args) metric_handle = args[next_idx];
            if (kwargs.contains("radius")) radius = kwargs["radius"].cast<double>();
            if (kwargs.contains("metric")) metric_handle = kwargs["metric"];

            if (metric_handle.is_none())
                throw py::value_error("Metric must be specified for find_all_within");

            if (py::isinstance<L1>(metric_handle))
                return self.template find_all_within<L1>(p, metric_handle.cast<L1>(), radius);
            if (py::isinstance<L2>(metric_handle))
                return self.template find_all_within<L2>(p, metric_handle.cast<L2>(), radius);
            if (py::isinstance<L2sq>(metric_handle))
                return self.template find_all_within<L2sq>(p, metric_handle.cast<L2sq>(), radius);
            if (py::isinstance<Linf>(metric_handle))
                return self.template find_all_within<Linf>(p, metric_handle.cast<Linf>(), radius);
            if (py::isinstance<Toroidal<L1, double>>(metric_handle))
                return self.template find_all_within<Toroidal<L1, double>>(p, metric_handle.cast<Toroidal<L1, double>>(), radius);
            if (py::isinstance<Toroidal<L2, double>>(metric_handle))
                return self.template find_all_within<Toroidal<L2, double>>(p, metric_handle.cast<Toroidal<L2, double>>(), radius);
            if (py::isinstance<Toroidal<L2sq, double>>(metric_handle))
                return self.template find_all_within<Toroidal<L2sq, double>>(p, metric_handle.cast<Toroidal<L2sq, double>>(), radius);
            if (py::isinstance<Toroidal<Linf, double>>(metric_handle))
                return self.template find_all_within<Toroidal<Linf, double>>(p, metric_handle.cast<Toroidal<Linf, double>>(), radius);
            if (py::isinstance<GreatCircle>(metric_handle))
                return self.template find_all_within<GreatCircle>(p, metric_handle.cast<GreatCircle>(), radius);

            throw py::type_error("Unsupported metric type");
        })

        // pop_closest supporting (point), (x,y) + metric/kwargs
        .def("pop_closest", [](Tree& self, py::args args, py::kwargs kwargs) -> Entry {
            PointType p;
            py::handle metric_handle = py::none();

            size_t n_args = args.size();
            size_t next_idx = 0;
            try {
                p = to_point<PointType>(args[0]);
                next_idx = 1;
            } catch (...) {
                if (n_args < 2) throw;
                p = PointType(args[0].cast<T>(), args[1].cast<T>());
                next_idx = 2;
            }

            if (next_idx < n_args) metric_handle = args[next_idx];
            if (kwargs.contains("metric")) metric_handle = kwargs["metric"];

            if (metric_handle.is_none() || py::isinstance<L2sq>(metric_handle))
                return self.template pop_closest<L2sq>(p, metric_handle.is_none() ? L2sq{} : metric_handle.cast<L2sq>());
            if (py::isinstance<L2>(metric_handle))
                return self.template pop_closest<L2>(p, metric_handle.cast<L2>());
            if (py::isinstance<L1>(metric_handle))
                return self.template pop_closest<L1>(p, metric_handle.cast<L1>());
            if (py::isinstance<Linf>(metric_handle))
                return self.template pop_closest<Linf>(p, metric_handle.cast<Linf>());
            if (py::isinstance<Toroidal<L1, double>>(metric_handle))
                return self.template pop_closest<Toroidal<L1, double>>(p, metric_handle.cast<Toroidal<L1, double>>());
            if (py::isinstance<Toroidal<L2, double>>(metric_handle))
                return self.template pop_closest<Toroidal<L2, double>>(p, metric_handle.cast<Toroidal<L2, double>>());
            if (py::isinstance<Toroidal<L2sq, double>>(metric_handle))
                return self.template pop_closest<Toroidal<L2sq, double>>(p, metric_handle.cast<Toroidal<L2sq, double>>());
            if (py::isinstance<Toroidal<Linf, double>>(metric_handle))
                return self.template pop_closest<Toroidal<Linf, double>>(p, metric_handle.cast<Toroidal<Linf, double>>());
            if (py::isinstance<GreatCircle>(metric_handle))
                return self.template pop_closest<GreatCircle>(p, metric_handle.cast<GreatCircle>());

            throw py::type_error("Unsupported metric type");
        })

        .def("rebalance", &Tree::rebalance)
        .def("balance_str", &Tree::balance_str)
        .def("depth_max", static_cast<size_t (Tree::*)() const>(&Tree::depth_max))
        .def("depth_avg", &Tree::depth_avg)
        .def("depth_stddev", &Tree::depth_stddev)
        .def("balance_factor", &Tree::balance_factor)
        .def("__len__", &Tree::size)
        .def("__bool__", [](const Tree& t) { return !t.empty(); })
        .def("__repr__", &Tree::balance_str)
        .def("__iter__", [](const Tree& t) {
            return py::make_iterator(t.begin(), t.end());
        }, py::keep_alive<0, 1>());
}

template<typename T>
void bind_point(py::module_& m, const std::string& name) {
    using P = Point<T>;

    py::class_<P>(m, name.c_str())
        .def(py::init<>())
        .def(py::init<T, T>())
        .def(py::init([](py::tuple t) {
            if (t.size() != 2) throw std::runtime_error("Point requires 2 elements");
            return P(t[0].cast<T>(), t[1].cast<T>());
        }))
        .def(py::init([](py::list l) {
            if (l.size() != 2) throw std::runtime_error("Point requires 2 elements");
            return P(l[0].cast<T>(), l[1].cast<T>());
        }))
        .def_readwrite("x", &P::x)
        .def_readwrite("y", &P::y)
        .def("__repr__", [](const P& p) {
            return "{" + std::to_string(p.x) + ", " + std::to_string(p.y) + "}";
        })
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def("__getitem__", [](const P& p, int i) -> T {
            if (i < 0 || i > 1) throw py::index_error();
            return p.coords[i];
        })
        .def("__setitem__", [](P& p, int i, T val) {
            if (i < 0 || i > 1) throw py::index_error();
            p.coords[i] = val;
        })
        .def("__len__", [](const P&) { return 2; });
}

template<typename PointType, typename ValueType>
void bind_entry(py::module_& m, const std::string& name) {
    using E = Entry<PointType, ValueType>;

    py::class_<E>(m, name.c_str())
        .def(py::init<>())
        .def(py::init<PointType, ValueType>())
        .def_readwrite("p", &E::p)
        .def_readwrite("value", &E::value)
        .def("__repr__", [](const E& e) {
            std::ostringstream oss;
            oss << "Entry(" << e.p << ", " << e.value << ")";
            return oss.str();
        })
        .def(py::self == py::self)
        .def(py::self != py::self);
}

PYBIND11_MODULE(kdtree, m) {
    m.doc() = R"doc(
        KDTree: Dynamic 2D spatial index
    )doc";

    // Bind Metric policies
    py::class_<L1>(m, "L1").def(py::init<>());
    py::class_<L2>(m, "L2").def(py::init<>());
    py::class_<L2sq>(m, "L2sq").def(py::init<>());
    py::class_<Linf>(m, "Linf").def(py::init<>());

    py::class_<GreatCircle>(m, "GreatCircle")
        .def(py::init<double>(), py::arg("radius") = 6371000.0)
        .def_readwrite("radius", &GreatCircle::radius);

    // Bind Toroidal adapter (pre-bound for common metrics)
    py::class_<Toroidal<L1, double>>(m, "ToroidalL1")
        .def(py::init<Pointd>(), py::arg("bounds"));
    py::class_<Toroidal<L2, double>>(m, "ToroidalL2")
        .def(py::init<Pointd>(), py::arg("bounds"));
    py::class_<Toroidal<L2sq, double>>(m, "ToroidalL2sq")
        .def(py::init<Pointd>(), py::arg("bounds"));
    py::class_<Toroidal<Linf, double>>(m, "ToroidalLinf")
        .def(py::init<Pointd>(), py::arg("bounds"));

    // Bind Point types (only int and double for Python)
    bind_point<int>(m, "Pointi");
    bind_point<double>(m, "Pointd");

    // Bind Entry types for type annotations
    bind_entry<Pointi, int64_t>(m, "Entryi");
    bind_entry<Pointd, int64_t>(m, "Entryd");
    bind_entry<Pointi, py::object>(m, "EntryPyi");
    bind_entry<Pointd, py::object>(m, "EntryPyd");

    // Generic "Entry" alias for documentation
    m.attr("Entry") = m.attr("Entryd");

    // Bind int64_t storage trees (for indices/IDs)
    bind_kdtree<Pointi, int64_t>(m, "KDTreei");
    bind_kdtree<Pointd, int64_t>(m, "KDTreed");

    // Bind Python object storage trees (for arbitrary objects)
    bind_kdtree<Pointi, py::object>(m, "KDTreePyi");
    bind_kdtree<Pointd, py::object>(m, "KDTreePyd");

    m.attr("__version__") = "1.0.0";
}
