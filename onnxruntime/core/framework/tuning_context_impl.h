// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file contains the implementation of TuningContext. At the moment, there is no necessity to expose these
// methods as OrtApis. This will cause missing symbols when loading provider dynamic libraries, because the libraries
// are not whole-archive linked and these symbols are not referenced at framework level. To circumvent this problem,
// the EP must has and only has one translation unit include this file.
#ifndef TUNING_CONTEXT_IMPL
#error define TUNING_CONTEXT_IMPL to use this header (impl) file
#endif

#pragma once

#include <functional>
#include <unordered_set>
#include <utility>

#include "core/framework/tunable.h"
#include "core/framework/tuning_context.h"
#include "core/framework/tuning_results.h"

namespace onnxruntime {

KernelMap TuningResultsManager::Lookup(const std::string& op_signature) const {
  std::scoped_lock l{lock_};
  auto it = results_.find(op_signature);
  if (it == results_.cend()) {
    return {};
  }
  return it->second;  // copied
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
int TuningResultsManager::Lookup(const std::string& op_signature, const std::string& params_signature) const {
  std::scoped_lock l{lock_};
  auto kernel_map_it = results_.find(op_signature);
  if (kernel_map_it == results_.cend()) {
    return -1;
  }

  const auto& km = kernel_map_it->second;
  auto it = km.find(params_signature);
  if (it == km.cend()) {
    return -1;
  }
  return it->second;
}

inline void AddImpl(const std::string& op_signature,
                    const std::string& params_signature,
                    int best_id,
                    KernelMap& kernel_map) {
  auto it = kernel_map.find(params_signature);
  if (it != kernel_map.end()) {
    if (it->second != best_id) {
      LOGS_DEFAULT(WARNING) << op_signature << "(" << params_signature << ") already have a best kernel "
                            << "id=" << it->second << " selected, want to add a different best kernel id=" << best_id
                            << ", the new kernel id will be ignored.";
    }
    return;
  }

  kernel_map[params_signature] = best_id;
}

void TuningResultsManager::Add(const std::string& op_signature, const std::string& params_signature, int best_id) {
  std::scoped_lock l{lock_};

  auto it = results_.find(op_signature);
  if (it == results_.end()) {
    it = results_.insert({op_signature, {}}).first;
  }

  AddImpl(op_signature, params_signature, best_id, it->second);
}

std::unordered_map<std::string, KernelMap> TuningResultsManager::Dump() const {
  std::scoped_lock l{lock_};
  return results_;
}

void MergeImpl(
    const std::string& op_signature,
    const KernelMap& kernel_map,
    /*out*/ std::unordered_map<std::string, KernelMap>& results) {
  auto it = results.find(op_signature);
  if (it == results.end()) {
    results[op_signature] = kernel_map;
    return;
  }

  for (const auto& [params_signature, best_id] : kernel_map) {
    AddImpl(op_signature, params_signature, best_id, it->second);
  }
}

void TuningResultsManager::Load(const std::unordered_map<std::string, KernelMap>& results_to_load) {
  std::scoped_lock l{lock_};
  for (const auto& [op_signature, kernel_map] : results_to_load) {
    MergeImpl(op_signature, kernel_map, results_);
  }
}

void TuningResultsManager::Merge(const std::string& op_signature, const KernelMap& kernel_map) {
  std::scoped_lock l{lock_};
  MergeImpl(op_signature, kernel_map, results_);
}

void TuningResultsManager::Clear() {
  results_ = {};
}

bool CheckMandatoryKeys(
    const TuningResultsValidator::CheckWriteFuncs& check_write_funcs,
    const std::unordered_map<std::string, std::string>& to_check) {
  constexpr const std::array mandatory_keys{"ORT_VERSION", "ORT_GIT_COMMIT", "ORT_BUILD_CONFIG"};

  bool passed = true;
  for (const auto& k : mandatory_keys) {
    if (check_write_funcs.find(k) == check_write_funcs.end()) {
      passed = false;
      LOGS_DEFAULT(ERROR) << "key=\"" << k << "\" is not registered for Check and Write.";
    }

    if (to_check.find(k) == to_check.end()) {
      passed = false;
      LOGS_DEFAULT(ERROR) << "key=\"" << k << "\" is not provided for validation.";
    }
  }

  return passed;
}

bool CheckKeysMatching(
    const TuningResultsValidator::CheckWriteFuncs& cw_funcs,
    const std::unordered_map<std::string, std::string>& to_check) {
  auto get_keys = [](const auto& it) -> std::string { return it.first; };
  std::unordered_set<std::string> required_keys;
  std::unordered_set<std::string> provided_keys;
  std::transform(cw_funcs.cbegin(), cw_funcs.cend(), std::inserter(required_keys, required_keys.end()), get_keys);
  std::transform(to_check.cbegin(), to_check.cend(), std::inserter(provided_keys, provided_keys.end()), get_keys);

  std::unordered_set<std::string> intersection;
  std::set_intersection(required_keys.cbegin(), required_keys.cend(),
                        provided_keys.cbegin(), provided_keys.cend(),
                        std::inserter(intersection, intersection.end()));
  bool matched = true;
  if (intersection.size() != required_keys.size()) {
    matched = false;
    for (const auto& k : required_keys) {
      if (intersection.find(k) == intersection.end()) {
        LOGS_DEFAULT(ERROR)
            << "Unmatched validator: \"" << k << "\" is required, but the tuning results does not provide it.";
      }
    }
  }
  if (intersection.size() != provided_keys.size()) {
    matched = false;
    for (const auto& k : provided_keys) {
      if (intersection.find(k) == intersection.end()) {
        LOGS_DEFAULT(ERROR)
            << "Unmatched validator: \"" << k << "\" is provided, but onnxruntime is unable to consume it.";
      }
    }
  }

  return matched;
}

Status TuningResultsValidator::CheckOrtVersion(const std::string& value) const {
  ORT_RETURN_IF(value != ORT_VERSION, "onnxruntime version mismatch");
  return Status::OK();
}

std::string TuningResultsValidator::WriteOrtVersion() const {
  return ORT_VERSION;
}

Status TuningResultsValidator::CheckOrtGitCommit(const std::string& value) const {
  // TODO:
  ORT_UNUSED_PARAMETER(value);
  return Status::OK();
}

std::string TuningResultsValidator::WriteOrtGitCommit() const {
  // TODO:
  return "";
}

Status TuningResultsValidator::CheckOrtBuildConfig(const std::string& value) const {
  auto current = WriteOrtBuildConfig();
  ORT_RETURN_IF(current != value,
                "onnxruntime building configuration mismatch: tuning results produced with library \"",
                value, "\", current library built with \"", current, "\"");
  return Status::OK();
}

std::string TuningResultsValidator::WriteOrtBuildConfig() const {
  return "";
}

TuningResultsValidator::TuningResultsValidator() {
  RegisterValidator(
      "ORT_VERSION",
      [this](auto&& k) { return CheckOrtVersion(std::forward<decltype(k)>(k)); },
      [this]() { return WriteOrtVersion(); });

  RegisterValidator(
      "ORT_GIT_COMMIT",
      [this](auto&& k) { return CheckOrtGitCommit(std::forward<decltype(k)>(k)); },
      [this]() { return WriteOrtGitCommit(); });

  RegisterValidator(
      "ORT_BUILD_CONFIG",
      [this](auto&& k) { return CheckOrtBuildConfig(std::forward<decltype(k)>(k)); },
      [this]() { return WriteOrtBuildConfig(); });
}

Status TuningResultsValidator::CheckAll(const std::unordered_map<std::string, std::string>& to_check) const {
  bool have_mandatory_keys = CheckMandatoryKeys(validators_, to_check);
  bool key_matched = CheckKeysMatching(validators_, to_check);
  ORT_RETURN_IF(!have_mandatory_keys || !key_matched,
                "An error occurs during the loading of tuning results. Check logs for more details.");

  for (const auto& [key, value] : to_check) {
    const auto& it = validators_.find(key);
    ORT_ENFORCE(it != validators_.cend());
    const CheckFunc& checker = it->second.first;
    ORT_RETURN_IF_ERROR(checker(value));
  }

  return Status::OK();
}

std::unordered_map<std::string, std::string> TuningResultsValidator::WriteAll() const {
  std::unordered_map<std::string, std::string> ret;
  for (const auto& [key, check_write_func_pair] : validators_) {
    const WriteFunc& writer = check_write_func_pair.second;
    ret[key] = writer();
  }
  return ret;
}

void TuningResultsValidator::RegisterValidator(const std::string& key, const CheckFunc& cf, const WriteFunc& wf) {
  ORT_ENFORCE(validators_.find(key) == validators_.end());
  validators_[key] = std::make_pair(cf, wf);
}

}  // namespace onnxruntime
