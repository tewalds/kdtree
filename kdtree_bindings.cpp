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
    // Accept other point variants (e.g. passing Pointi to a Pointd tree)
    if (py::isinstance<Pointi>(obj)) {
        auto p = obj.cast<Pointi>();
        return PointType(static_cast<T>(p.x), static_cast<T>(p.y));
    }
    if (py::isinstance<Pointd>(obj)) {
        auto p = obj.cast<Pointd>();
        return PointType(static_cast<T>(p.x), static_cast<T>(p.y));
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
        auto seq = obj.cast<py::sequence>();
        return PointType(seq[0].cast<T>(), seq[1].cast<T>());
    }

    throw std::runtime_error("Cannot convert to Point - expected Point, tuple, list, or 1D buffer of size 2");
}

// Helper to dispatch to the correct metric type and execute a function
template<typename Func>
auto dispatch_metric(py::handle handle, Func&& f) {
    if (handle.is_none() || py::isinstance<L2sq>(handle))
        return f(handle.is_none() ? L2sq{} : handle.cast<L2sq>());
    if (py::isinstance<L1>(handle))
        return f(handle.cast<L1>());
    if (py::isinstance<L2>(handle))
        return f(handle.cast<L2>());
    if (py::isinstance<Linf>(handle))
        return f(handle.cast<Linf>());
    if (py::isinstance<Toroidal<L1, double>>(handle))
        return f(handle.cast<Toroidal<L1, double>>());
    if (py::isinstance<Toroidal<L2, double>>(handle))
        return f(handle.cast<Toroidal<L2, double>>());
    if (py::isinstance<Toroidal<L2sq, double>>(handle))
        return f(handle.cast<Toroidal<L2sq, double>>());
    if (py::isinstance<Toroidal<Linf, double>>(handle))
        return f(handle.cast<Toroidal<Linf, double>>());
    if (py::isinstance<GreatCircle>(handle))
        return f(handle.cast<GreatCircle>());
    throw py::type_error("Unsupported metric type");
}

// Helper to add dist() method to metric classes
template<typename Metric>
void add_dist(py::class_<Metric>& cl) {
    cl.def("dist", [](const Metric& self, py::handle a, py::handle b) {
        return self.dist(to_point<Pointd>(a), to_point<Pointd>(b));
    }, py::arg("a"), py::arg("b"));
}

template<typename PointType, typename ValueType>
void bind_kdtree(py::module_& m, const std::string& name) {
    using Tree = KDTree<PointType, ValueType>;
    using Entry = typename Tree::Entry;
    using T = typename PointType::value_type;

    py::class_<Tree>(m, name.c_str())
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
                py::buffer_info val_info = values.cast<py::buffer>().request();
                if (val_info.size != (ssize_t)n) {
                    throw std::runtime_error("Values length must match coordinates");
                }
                const ValueType* val_ptr = static_cast<const ValueType*>(val_info.ptr);
                for (size_t i = 0; i < n; ++i) {
                    entries.push_back({PointType(coords_ptr[i*2], coords_ptr[i*2+1]), val_ptr[i]});
                }
            } else {
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

        .def("insert", [](Tree& self, py::handle point_like, ValueType val) {
            return self.insert(to_point<PointType>(point_like), val);
        }, py::arg("point"), py::arg("value"))
        .def("remove", [](Tree& self, py::handle point_like) {
            return self.remove(to_point<PointType>(point_like));
        }, py::arg("point"))
        .def("exists", [](const Tree& self, py::handle point_like) {
            return self.exists(to_point<PointType>(point_like));
        }, py::arg("point"))
        .def("find", [](const Tree& self, py::handle point_like) {
            return self.find(to_point<PointType>(point_like));
        }, py::arg("point"))

        .def("find_closest", [](const Tree& self, py::handle point_like, py::handle metric, py::handle max_dist) -> std::optional<Entry> {
            double limit = max_dist.is_none() ? -1.0 : max_dist.cast<double>();
            if (limit >= 0 && metric.is_none())
                throw py::value_error("Metric must be specified when max_dist is provided");
            return dispatch_metric(metric, [&](const auto& m) -> std::optional<Entry> {
                return self.template find_closest(to_point<PointType>(point_like), m, limit);
            });
        }, py::arg("point"), py::arg("metric") = py::none(), py::arg("max_dist") = py::none())

        .def("find_closest_k", [](const Tree& self, py::handle point_like, size_t k, py::handle metric, py::handle max_dist) -> std::vector<Entry> {
            double limit = max_dist.is_none() ? -1.0 : max_dist.cast<double>();
            if (limit >= 0 && metric.is_none())
                throw py::value_error("Metric must be specified when max_dist is provided");
            return dispatch_metric(metric, [&](const auto& m) -> std::vector<Entry> {
                return self.template find_closest_k(to_point<PointType>(point_like), k, m, limit);
            });
        }, py::arg("point"), py::arg("k") = 1, py::arg("metric") = py::none(), py::arg("max_dist") = py::none())

        .def("find_all_within", [](const Tree& self, py::handle point_like, py::handle metric, double radius) -> std::vector<Entry> {
            if (metric.is_none())
                throw py::value_error("Metric must be specified for find_all_within");
            return dispatch_metric(metric, [&](const auto& m) -> std::vector<Entry> {
                return self.template find_all_within(to_point<PointType>(point_like), m, radius);
            });
        }, py::arg("point"), py::arg("metric"), py::arg("radius"))

        .def("pop_closest", [](Tree& self, py::handle point_like, py::handle metric, py::handle max_dist) -> std::optional<Entry> {
            double limit = max_dist.is_none() ? -1.0 : max_dist.cast<double>();
            return dispatch_metric(metric, [&](const auto& m) -> std::optional<Entry> {
                return self.template pop_closest(to_point<PointType>(point_like), m, limit);
            });
        }, py::arg("point"), py::arg("metric") = py::none(), py::arg("max_dist") = py::none())

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

    auto l1 = py::class_<L1>(m, "L1").def(py::init<>()); add_dist(l1);
    auto l2 = py::class_<L2>(m, "L2").def(py::init<>()); add_dist(l2);
    auto l2sq = py::class_<L2sq>(m, "L2sq").def(py::init<>()); add_dist(l2sq);
    auto linf = py::class_<Linf>(m, "Linf").def(py::init<>()); add_dist(linf);

    auto gc = py::class_<GreatCircle>(m, "GreatCircle")
        .def(py::init<double>(), py::arg("radius") = 6371000.0)
        .def_readwrite("radius", &GreatCircle::radius);
    add_dist(gc);

    auto tl1 = py::class_<Toroidal<L1, double>>(m, "ToroidalL1").def(py::init<Pointd>(), py::arg("bounds")); add_dist(tl1);
    auto tl2 = py::class_<Toroidal<L2, double>>(m, "ToroidalL2").def(py::init<Pointd>(), py::arg("bounds")); add_dist(tl2);
    auto tl2sq = py::class_<Toroidal<L2sq, double>>(m, "ToroidalL2sq").def(py::init<Pointd>(), py::arg("bounds")); add_dist(tl2sq);
    auto tlinf = py::class_<Toroidal<Linf, double>>(m, "ToroidalLinf").def(py::init<Pointd>(), py::arg("bounds")); add_dist(tlinf);

    bind_point<int>(m, "Pointi");
    bind_point<double>(m, "Pointd");

    bind_entry<Pointi, int64_t>(m, "Entryi");
    bind_entry<Pointd, int64_t>(m, "Entryd");
    bind_entry<Pointi, py::object>(m, "EntryPyi");
    bind_entry<Pointd, py::object>(m, "EntryPyd");

    bind_kdtree<Pointi, int64_t>(m, "KDTreei");
    bind_kdtree<Pointd, int64_t>(m, "KDTreed");

    bind_kdtree<Pointi, py::object>(m, "KDTreePyi");
    bind_kdtree<Pointd, py::object>(m, "KDTreePyd");

    m.attr("__version__") = "1.0.0";
}
