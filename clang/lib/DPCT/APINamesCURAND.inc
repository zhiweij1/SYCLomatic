//===--------------- APINamesCURAND.inc -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

CALL_FACTORY_ENTRY("skipahead",
                   CALL("oneapi::mkl::rng::device::skip_ahead",
                        MEMBER_CALL(DEREF(makeDerefArgCreatorWithCall(1)),
                                    false, "get_engine"),
                        ARG(0)))

CONDITIONAL_FACTORY_ENTRY(
NeedExtraParens(0),
CONDITIONAL_FACTORY_ENTRY(
    CheckArgType(1, "struct curandStateMRG32k3a *"),
        CALL_FACTORY_ENTRY(
            "skipahead_sequence",
            CALL("oneapi::mkl::rng::device::skip_ahead",
                 MEMBER_CALL(DEREF(makeDerefArgCreatorWithCall(1)),
                                        false, "get_engine"),
                 makeCombinedArg(makeCombinedArg(ARG("{0, ("), ARG(0)),
                                 ARG(") * (std::uint64_t(1) << 63)}")))),
    CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "struct curandStatePhilox4_32_10 *"),
            CALL_FACTORY_ENTRY(
                "skipahead_sequence",
                CALL("oneapi::mkl::rng::device::skip_ahead",
                     MEMBER_CALL(DEREF(makeDerefArgCreatorWithCall(1)),
                                        false, "get_engine"),
                     makeCombinedArg(
                         makeCombinedArg(ARG("{0, static_cast<std::uint64_t>(("),
                                         ARG(0)),
                         ARG(") * 4)}")))),
        REMOVE_API_FACTORY_ENTRY_WITH_MSG("skipahead_sequence", "this API is not supported for mcg59 engine."))),
CONDITIONAL_FACTORY_ENTRY(
    CheckArgType(1, "struct curandStateMRG32k3a *"),
        CALL_FACTORY_ENTRY(
            "skipahead_sequence",
            CALL("oneapi::mkl::rng::device::skip_ahead",
                 MEMBER_CALL(DEREF(makeDerefArgCreatorWithCall(1)),
                                        false, "get_engine"),
                 makeCombinedArg(makeCombinedArg(ARG("{0, "), ARG(0)),
                                 ARG(" * (std::uint64_t(1) << 63)}")))),
    CONDITIONAL_FACTORY_ENTRY(
        CheckArgType(1, "struct curandStatePhilox4_32_10 *"),
            CALL_FACTORY_ENTRY(
                "skipahead_sequence",
                CALL("oneapi::mkl::rng::device::skip_ahead",
                     MEMBER_CALL(DEREF(makeDerefArgCreatorWithCall(1)),
                                        false, "get_engine"),
                     makeCombinedArg(
                         makeCombinedArg(ARG("{0, static_cast<std::uint64_t>("),
                                         ARG(0)),
                         ARG(" * 4)}")))),
        REMOVE_API_FACTORY_ENTRY_WITH_MSG("skipahead_sequence", "this API is not supported for mcg59 engine.")))
)

CONDITIONAL_FACTORY_ENTRY(
NeedExtraParens(0),
    CALL_FACTORY_ENTRY(
        "skipahead_subsequence",
        CALL("oneapi::mkl::rng::device::skip_ahead",
             MEMBER_CALL(DEREF(makeDerefArgCreatorWithCall(1)),
                                        false, "get_engine"),
             makeCombinedArg(
                 makeCombinedArg(ARG("{0, static_cast<std::uint64_t>(("),
                                 ARG(0)),
                 ARG(") * 8)}")))),
    CALL_FACTORY_ENTRY(
        "skipahead_subsequence",
        CALL("oneapi::mkl::rng::device::skip_ahead",
             MEMBER_CALL(DEREF(makeDerefArgCreatorWithCall(1)),
                                        false, "get_engine"),
             makeCombinedArg(
                 makeCombinedArg(ARG("{0, static_cast<std::uint64_t>("),
                                 ARG(0)),
                 ARG(" * 8)}"))))
)

// bits
CONDITIONAL_FACTORY_ENTRY(
    CheckArgType(0, "struct curandStateXORWOW *"),
    MEMBER_CALL_FACTORY_ENTRY("curand",
                              DEREF(makeDerefArgCreatorWithCall(0)), false,
                              "generate<oneapi::mkl::rng::device::uniform_"
                              "bits<std::uint32_t>, 1>"),
    MEMBER_CALL_FACTORY_ENTRY(
        "curand", DEREF(makeDerefArgCreatorWithCall(0)), false,
        "generate<oneapi::mkl::rng::device::bits<std::uint32_t>, 1>"))
MEMBER_CALL_FACTORY_ENTRY("curand4",
                          DEREF(makeDerefArgCreatorWithCall(0)), false,
                          "generate<oneapi::mkl::rng::device::bits<std::uint32_t>, 4>")
// gaussian
MEMBER_CALL_FACTORY_ENTRY("curand_normal",
                          DEREF(makeDerefArgCreatorWithCall(0)), false,
                          "generate<oneapi::mkl::rng::device::gaussian<float>, 1>")
MEMBER_CALL_FACTORY_ENTRY("curand_normal2",
                          DEREF(makeDerefArgCreatorWithCall(0)), false,
                          "generate<oneapi::mkl::rng::device::gaussian<float>, 2>")
MEMBER_CALL_FACTORY_ENTRY("curand_normal2_double",
                          DEREF(makeDerefArgCreatorWithCall(0)), false,
                          "generate<oneapi::mkl::rng::device::gaussian<double>, 2>")
MEMBER_CALL_FACTORY_ENTRY("curand_normal4",
                          DEREF(makeDerefArgCreatorWithCall(0)), false,
                          "generate<oneapi::mkl::rng::device::gaussian<float>, 4>")
MEMBER_CALL_FACTORY_ENTRY("curand_normal_double",
                          DEREF(makeDerefArgCreatorWithCall(0)), false,
                          "generate<oneapi::mkl::rng::device::gaussian<double>, 1>")
MEMBER_CALL_FACTORY_ENTRY("curand_normal4_double",
                          DEREF(makeDerefArgCreatorWithCall(0)), false,
                          "generate<oneapi::mkl::rng::device::gaussian<double>, 4>")
// lognormal
MEMBER_CALL_FACTORY_ENTRY("curand_log_normal",
                          DEREF(makeDerefArgCreatorWithCall(0)), false,
                          "generate<oneapi::mkl::rng::device::lognormal<float>, 1>", ARG(1), ARG(2))
MEMBER_CALL_FACTORY_ENTRY("curand_log_normal2",
                          DEREF(makeDerefArgCreatorWithCall(0)), false,
                          "generate<oneapi::mkl::rng::device::lognormal<float>, 2>", ARG(1), ARG(2))
MEMBER_CALL_FACTORY_ENTRY("curand_log_normal2_double",
                          DEREF(makeDerefArgCreatorWithCall(0)), false,
                          "generate<oneapi::mkl::rng::device::lognormal<double>, 2>", ARG(1), ARG(2))
MEMBER_CALL_FACTORY_ENTRY("curand_log_normal4",
                          DEREF(makeDerefArgCreatorWithCall(0)), false,
                          "generate<oneapi::mkl::rng::device::lognormal<float>, 4>", ARG(1), ARG(2))
MEMBER_CALL_FACTORY_ENTRY("curand_log_normal_double",
                          DEREF(makeDerefArgCreatorWithCall(0)), false,
                          "generate<oneapi::mkl::rng::device::lognormal<double>, 1>", ARG(1), ARG(2))
MEMBER_CALL_FACTORY_ENTRY("curand_log_normal4_double",
                          DEREF(makeDerefArgCreatorWithCall(0)), false,
                          "generate<oneapi::mkl::rng::device::lognormal<double>, 4>", ARG(1), ARG(2))
// uniform
MEMBER_CALL_FACTORY_ENTRY("curand_uniform",
                          DEREF(makeDerefArgCreatorWithCall(0)), false,
                          "generate<oneapi::mkl::rng::device::uniform<float>, 1>")
MEMBER_CALL_FACTORY_ENTRY("curand_uniform2_double",
                          DEREF(makeDerefArgCreatorWithCall(0)), false,
                          "generate<oneapi::mkl::rng::device::uniform<double>, 2>")
MEMBER_CALL_FACTORY_ENTRY("curand_uniform4",
                          DEREF(makeDerefArgCreatorWithCall(0)), false,
                          "generate<oneapi::mkl::rng::device::uniform<float>, 4>")
MEMBER_CALL_FACTORY_ENTRY("curand_uniform_double",
                          DEREF(makeDerefArgCreatorWithCall(0)), false,
                          "generate<oneapi::mkl::rng::device::uniform<double>, 1>")
MEMBER_CALL_FACTORY_ENTRY("curand_uniform4_double",
                          DEREF(makeDerefArgCreatorWithCall(0)), false,
                          "generate<oneapi::mkl::rng::device::uniform<double>, 4>")
// Poisson
MEMBER_CALL_FACTORY_ENTRY("curand_poisson",
                          DEREF(makeDerefArgCreatorWithCall(0)), false,
                          "generate<oneapi::mkl::rng::device::poisson<std::uint32_t>, 1>", ARG(1))
MEMBER_CALL_FACTORY_ENTRY("curand_poisson4",
                          DEREF(makeDerefArgCreatorWithCall(0)), false,
                          "generate<oneapi::mkl::rng::device::poisson<std::uint32_t>, 4>", ARG(1))
