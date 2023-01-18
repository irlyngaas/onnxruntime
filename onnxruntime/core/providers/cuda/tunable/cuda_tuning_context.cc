// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tunable/cuda_tuning_context.h"

#include "core/providers/shared_library/provider_api.h"
#include "core/framework/tuning_context.h"
#define TUNING_CONTEXT_IMPL
#include "core/framework/tuning_context_impl.h"
#undef TUNING_CONTEXT_IMPL
#include <core/providers/cuda/cuda_execution_provider.h>

namespace onnxruntime {
namespace cuda {
namespace tunable {

std::string WriteCudaVersion() {
  int version;
  CUDA_CALL_THROW(cudaRuntimeGetVersion(&version));
  return std::to_string(version);
}

Status CheckCudaVersion(const std::string& value) {
  auto current = WriteCudaVersion();
  ORT_RETURN_IF(current != value, "CUDA runtime version mismatch: tuning results produced with CUDA ", value,
                ", onnxruntime currently run with CUDA ", current);
  return Status::OK();
}

std::string CudaTuningResultsValidator::WriteDeviceModel() const {
  return ep_->GetDeviceProp().name;
}

Status CudaTuningResultsValidator::CheckDeviceModel(const std::string& value) const {
  auto current = WriteDeviceModel();
  ORT_RETURN_IF(current != value, "Device model mismatch: tuning results produced with device ", value,
                ", onnxruntime currently run with device ", current);
  return Status::OK();
}

CudaTuningResultsValidator::CudaTuningResultsValidator(CUDAExecutionProvider* ep) : ep_(ep) {
  RegisterValidator("CUDA_VERSION", CheckCudaVersion, WriteCudaVersion);
  RegisterValidator(
      "DEVICE_MODEL",
      [this](const std::string& value) { return CheckDeviceModel(value); },
      [this]() { return WriteDeviceModel(); });
}

CudaTuningContext::CudaTuningContext(CUDAExecutionProvider* ep, TunableOpInfo* info)
    : ITuningContext(ep), info_(info), validator_(ep) {}

void CudaTuningContext::EnableTunableOp() {
  LOGS_DEFAULT(INFO) << "Enable TunableOp for CUDA Execution Provider";
  info_->enabled = true;
}

void CudaTuningContext::DisableTunableOp() {
  LOGS_DEFAULT(INFO) << "Disable TunableOp for CUDA Execution Provider";
  info_->enabled = false;
}

bool CudaTuningContext::IsTunableOpEnabled() const {
  return info_->enabled;
}

TuningResultsManager& CudaTuningContext::GetTuningResultsManager() {
  return manager_;
}

const TuningResultsManager& CudaTuningContext::GetTuningResultsManager() const {
  return manager_;
}

const TuningResultsValidator& CudaTuningContext::GetTuningResultsValidator() const {
  return validator_;
}

}  // namespace tunable
}  // namespace cuda
}  // namespace onnxruntime
