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
    
    throw std::runtime_error("Cannot convert to Point - expected Point, tuple, or list");
}

template<typename ValueType, typename PointType>
void bind_kdtree(py::module_& m, const std::string& name) {
    using Tree = KDTree<ValueType, PointType>;
    using Value = typename Tree::Value;
    using T = typename PointType::value_type;
    
    auto tree_class = py::class_<Tree>(m, name.c_str())
        .def(py::init<>())
        .def(py::init<const std::vector<Value>&>())
        .def("empty", &Tree::empty)
        .def("size", &Tree::size)
        .def("clear", &Tree::clear)
        
        // insert() overloads - never spell out Value type!
        .def("insert", &Tree::insert, py::arg("value"))
        .def("insert", [](Tree& self, ValueType val, const PointType& p) {
            return self.insert(Value(val, p));
        }, py::arg("value"), py::arg("point"))
        .def("insert", [](Tree& self, ValueType val, py::handle point_like) {
            return self.insert(Value(val, to_point<PointType>(point_like)));
        }, py::arg("value"), py::arg("point"))
        .def("insert", [](Tree& self, ValueType val, T x, T y) {
            return self.insert(Value(val, PointType(x, y)));
        }, py::arg("value"), py::arg("x"), py::arg("y"))
        
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
        .def("find", static_cast<std::optional<Value> (Tree::*)(PointType) const>(&Tree::find), py::arg("point"))
        .def("find", [](const Tree& self, py::handle point_like) {
            return self.find(to_point<PointType>(point_like));
        }, py::arg("point"))
        .def("find", [](const Tree& self, T x, T y) {
            return self.find(PointType(x, y));
        }, py::arg("x"), py::arg("y"))
        
        // find_closest() with norm parameter
        .def("find_closest", [](const Tree& self, const PointType& p, Norm norm) {
            return self.find_closest(p, norm);
        }, py::arg("point"), py::arg("norm") = Norm::L2)
        .def("find_closest", [](const Tree& self, py::handle point_like, Norm norm) {
            return self.find_closest(to_point<PointType>(point_like), norm);
        }, py::arg("point"), py::arg("norm") = Norm::L2)
        .def("find_closest", [](const Tree& self, T x, T y, Norm norm) {
            return self.find_closest(PointType(x, y), norm);
        }, py::arg("x"), py::arg("y"), py::arg("norm") = Norm::L2)
        
        // pop_closest() with norm parameter
        .def("pop_closest", [](Tree& self, const PointType& p, Norm norm) {
            return self.pop_closest(p, norm);
        }, py::arg("point"), py::arg("norm") = Norm::L2)
        .def("pop_closest", [](Tree& self, py::handle point_like, Norm norm) {
            return self.pop_closest(to_point<PointType>(point_like), norm);
        }, py::arg("point"), py::arg("norm") = Norm::L2)
        .def("pop_closest", [](Tree& self, T x, T y, Norm norm) {
            return self.pop_closest(PointType(x, y), norm);
        }, py::arg("x"), py::arg("y"), py::arg("norm") = Norm::L2)
        
        .def("rebalance", &Tree::rebalance)
        .def("balance_str", &Tree::balance_str)
        .def("depth_max", &Tree::depth_max)
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
        .def("distance", &P::distance, py::arg("other"))
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

template<typename ValueType, typename PointType>
void bind_value(py::module_& m, const std::string& name) {
    using V = Value<ValueType, PointType>;
    
    py::class_<V>(m, name.c_str())
        .def(py::init<>())
        .def(py::init<ValueType, PointType>())
        .def_readwrite("value", &V::value)
        .def_readwrite("p", &V::p)
        .def("__repr__", [](const V& v) {
            std::ostringstream oss;
            oss << "Value(" << v.value << ", " << v.p << ")";
            return oss.str();
        })
        .def(py::self == py::self)
        .def(py::self != py::self);
}

PYBIND11_MODULE(kdtree, m) {
    m.doc() = R"doc(
        KDTree: Dynamic 2D spatial index
        
        Unlike scipy.spatial.KDTree, supports dynamic insert/remove without rebuilding.
        
        Tree types:
            KDTreei  - int coords, int64 values
            KDTreed  - double coords, int64 values (recommended)
            KDTreePyi - int coords, Python object values
            KDTreePyd - double coords, Python object values
        
        Example:
            tree = kdtree.KDTreed()
            tree.insert(42, (1.5, 2.3))
            tree.insert(7, 4.1, 3.7)
            result = tree.find_closest((2.0, 3.0))
            
            # Manhattan distance
            result_l1 = tree.find_closest((2.0, 3.0), kdtree.Norm.L1)
    )doc";
    
    py::enum_<Norm>(m, "Norm")
        .value("L1", Norm::L1, "Manhattan distance")
        .value("L2", Norm::L2, "Euclidean squared distance (default)")
        .export_values();
    
    // Bind Point types (only int and double for Python)
    bind_point<int>(m, "Pointi");
    bind_point<double>(m, "Pointd");
    
    // Bind Value types for type annotations
    bind_value<int64_t, Pointi>(m, "Valuei");
    bind_value<int64_t, Pointd>(m, "Valued");
    bind_value<py::object, Pointi>(m, "ValuePyi");
    bind_value<py::object, Pointd>(m, "ValuePyd");
    
    // Generic "Value" alias for documentation
    m.attr("Value") = m.attr("Valued");
    
    // Bind int64_t storage trees (for indices/IDs)
    bind_kdtree<int64_t, Pointi>(m, "KDTreei");
    bind_kdtree<int64_t, Pointd>(m, "KDTreed");
    
    // Bind Python object storage trees (for arbitrary objects)
    bind_kdtree<py::object, Pointi>(m, "KDTreePyi");
    bind_kdtree<py::object, Pointd>(m, "KDTreePyd");
    
    m.attr("__version__") = "1.0.0";
}
