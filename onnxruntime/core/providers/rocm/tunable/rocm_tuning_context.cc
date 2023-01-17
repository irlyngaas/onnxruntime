// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/tunable/rocm_tuning_context.h"

#include "core/providers/shared_library/provider_api.h"
#include "core/framework/tuning_context.h"
#define TUNING_CONTEXT_IMPL
#include "core/framework/tuning_context_impl.h"
#undef TUNING_CONTEXT_IMPL
#include "core/providers/rocm/rocm_execution_provider.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {

std::string WriteHipVersion() {
  int version;
  HIP_CALL_THROW(hipRuntimeGetVersion(&version));
  return std::to_string(version);
}

Status CheckHipVersion(const std::string& value) {
  auto current = WriteHipVersion();
  ORT_RETURN_IF(current != value, "HIP runtime version mismatch: tuning results produced with HIP ", value,
                ", onnxruntime currently run with HIP ", current);
  return Status::OK();
}

std::string WriteRocBlasVersion() {
  char buf[64];
  ROCBLAS_CALL_THROW(rocblas_get_version_string(buf, 256));
  buf[63] = '\0';
  return buf;
}

Status CheckRocBlasVersion(const std::string& value) {
  auto current = WriteRocBlasVersion();
  ORT_RETURN_IF(current != value, "rocblas runtime version mismatch: tuning results produced with rocblas ", value,
                ", onnxruntime currently run with rocblas ", current);
  return Status::OK();
}

std::string RocmTuningResultsValidator::WriteDeviceModel() const {
  return ep_->GetDeviceProp().name;
}

Status RocmTuningResultsValidator::CheckDeviceModel(const std::string& value) const {
  auto current = WriteDeviceModel();
  ORT_RETURN_IF(current != value, "Device model mismatch: tuning results produced with device ", value,
                ", onnxruntime currently run with device ", current);
  return Status::OK();
}

RocmTuningResultsValidator::RocmTuningResultsValidator(ROCMExecutionProvider* ep) : ep_{ep} {
  RegisterValidator("HIP_VERSION", CheckHipVersion, WriteHipVersion);
  RegisterValidator("ROCBLAS_VERSION", CheckRocBlasVersion, WriteRocBlasVersion);
  RegisterValidator(
      "DEVICE_MODEL",
      [this](const std::string& value) { return CheckDeviceModel(value); },
      [this]() { return WriteDeviceModel(); });
}

std::string RocmTuningResultsValidator::WriteOrtBuildConfig() const {
  std::ostringstream oss;
  oss << "USE_CK=" << USE_COMPOSABLE_KERNEL << "|";
#ifdef USE_ROCBLAS_EXTENSION_API
  oss << "USE_ROCBLAS_EXTENSION_API" << 1 << "|";
#else
  oss << "USE_ROCBLAS_EXTENSION_API" << 0 << "|";
#endif
  return oss.str();
}

RocmTuningContext::RocmTuningContext(ROCMExecutionProvider* ep, TunableOpInfo* info) : info_(info), validator_(ep) {}

void RocmTuningContext::EnableTunableOp() {
  LOGS_DEFAULT(INFO) << "Enable TunableOp for ROCm Execution Provider";
  info_->enabled = true;
}

void RocmTuningContext::DisableTunableOp() {
  LOGS_DEFAULT(INFO) << "Disable TunableOp for ROCm Execution Provider";
  info_->enabled = false;
}

bool RocmTuningContext::IsTunableOpEnabled() const {
  return info_->enabled;
}

TuningResultsManager& RocmTuningContext::GetTuningResultsManager() {
  return manager_;
}

const TuningResultsManager& RocmTuningContext::GetTuningResultsManager() const {
  return manager_;
}

const TuningResultsValidator& RocmTuningContext::GetTuningResultsValidator() const {
  return validator_;
}

}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
