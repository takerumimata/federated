load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_library(
    name = "jax",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow_federated/tools:__subpackages__"],
)

py_library(
    name = "mnist",
    srcs = ["mnist.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow_federated"],
)

py_test(
    name = "mnist_test",
    srcs = ["mnist_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":mnist",
        "//tensorflow_federated",
    ],
)
