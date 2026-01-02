// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers

// MARK: - Phase-Aware Download

/// Download the model with phase reporting.
///
/// This wraps the standard `downloadModel` function and emits loading phases
/// for UI feedback during the download process.
///
/// - Parameters:
///   - hub: HubApi instance
///   - configuration: the model identifier
///   - progressHandler: callback for download progress
///   - phaseHandler: callback for loading phase updates
/// - Returns: URL for the directory containing downloaded files
public func downloadModel(
    hub: HubApi, configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void,
    phaseHandler: @Sendable @escaping (LoadingPhase) -> Void
) async throws -> URL {
    phaseHandler(.downloadingConfig)

    do {
        switch configuration.id {
        case .id(let id, let revision):
            let repo = Hub.Repo(id: id)
            let modelFiles = ["*.safetensors", "*.json"]

            // Wrap progress handler to emit phase updates
            let wrappedProgressHandler: @Sendable (Progress) -> Void = { progress in
                progressHandler(progress)

                // Extract file info from progress if available
                if let currentFile = progress.userInfo[ProgressUserInfoKey("currentFile")] as? String,
                   let fileIndex = progress.userInfo[ProgressUserInfoKey("fileIndex")] as? Int,
                   let totalFiles = progress.userInfo[ProgressUserInfoKey("totalFiles")] as? Int {
                    phaseHandler(.downloadingWeights(
                        file: currentFile,
                        fileIndex: fileIndex,
                        totalFiles: totalFiles,
                        progress: progress.fractionCompleted
                    ))
                } else {
                    // Fallback: use progress fraction without file details
                    phaseHandler(.downloadingWeights(
                        file: "model files",
                        fileIndex: 1,
                        totalFiles: 1,
                        progress: progress.fractionCompleted
                    ))
                }
            }

            return try await hub.snapshot(
                from: repo,
                revision: revision,
                matching: modelFiles,
                progressHandler: wrappedProgressHandler
            )
        case .directory(let directory):
            return directory
        }

    } catch Hub.HubClientError.authorizationRequired {
        return configuration.modelDirectory(hub: hub)

    } catch {
        let nserror = error as NSError
        if nserror.domain == NSURLErrorDomain && nserror.code == NSURLErrorNotConnectedToInternet {
            return configuration.modelDirectory(hub: hub)
        } else {
            throw error
        }
    }
}

// MARK: - Phase-Aware Weight Loading

/// Load model weights with phase reporting.
///
/// This wraps the standard `loadWeights` function and emits loading phases
/// for each safetensor file being loaded.
///
/// - Parameters:
///   - modelDirectory: URL of the directory containing model files
///   - model: the language model to load weights into
///   - quantization: optional quantization configuration
///   - perLayerQuantization: optional per-layer quantization configuration
///   - phaseHandler: callback for loading phase updates
public func loadWeights(
    modelDirectory: URL, model: LanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    phaseHandler: @Sendable @escaping (LoadingPhase) -> Void
) throws {
    // Collect safetensor files first to know total count
    var safetensorURLs: [URL] = []
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" {
            safetensorURLs.append(url)
        }
    }

    let totalFiles = safetensorURLs.count
    var weights = [String: MLXArray]()

    // Load each file with phase reporting
    for (index, url) in safetensorURLs.enumerated() {
        let fileName = url.lastPathComponent
        phaseHandler(.loadingWeights(file: fileName, fileIndex: index + 1, totalFiles: totalFiles))

        let w = try loadArrays(url: url)
        for (key, value) in w {
            weights[key] = value
        }
    }

    phaseHandler(.initializingModel)

    // per-model cleanup
    weights = model.sanitize(weights: weights)

    // quantize if needed
    if quantization != nil || perLayerQuantization != nil {
        quantize(model: model) { path, module in
            if weights["\(path).scales"] != nil {
                if let perLayerQuantization {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                } else {
                    return quantization?.asTuple
                }
            } else {
                return nil
            }
        }
    }

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])

    eval(model)
}

// MARK: - Original Functions (Backward Compatible)

/// Download the model using the `HubApi`.
///
/// This will download `*.safetensors` and `*.json` if the ``ModelConfiguration``
/// represents a Hub id, e.g. `mlx-community/gemma-2-2b-it-4bit`.
///
/// This is typically called via ``ModelFactory/load(hub:configuration:progressHandler:)``
///
/// - Parameters:
///   - hub: HubApi instance
///   - configuration: the model identifier
///   - progressHandler: callback for progress
/// - Returns: URL for the directory containing downloaded files
public func downloadModel(
    hub: HubApi, configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void
) async throws -> URL {
    do {
        switch configuration.id {
        case .id(let id, let revision):
            // download the model weights
            let repo = Hub.Repo(id: id)
            let modelFiles = ["*.safetensors", "*.json"]
            return try await hub.snapshot(
                from: repo,
                revision: revision,
                matching: modelFiles,
                progressHandler: progressHandler
            )
        case .directory(let directory):
            return directory
        }

    } catch Hub.HubClientError.authorizationRequired {
        // an authorizationRequired means (typically) that the named repo doesn't exist on
        // on the server so retry with local only configuration
        return configuration.modelDirectory(hub: hub)

    } catch {
        let nserror = error as NSError
        if nserror.domain == NSURLErrorDomain && nserror.code == NSURLErrorNotConnectedToInternet {
            // Error Domain=NSURLErrorDomain Code=-1009 "The Internet connection appears to be offline."
            // fall back to the local directory
            return configuration.modelDirectory(hub: hub)
        } else {
            throw error
        }
    }
}

/// Load model weights.
///
/// This is typically called via ``ModelFactory/load(hub:configuration:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``LanguageModel/sanitize(weights:)``, applies optional quantization, and
/// updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: LanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
) throws {
    // load the weights
    var weights = [String: MLXArray]()
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" {
            let w = try loadArrays(url: url)
            for (key, value) in w {
                weights[key] = value
            }
        }
    }

    // per-model cleanup
    weights = model.sanitize(weights: weights)

    // quantize if needed
    if quantization != nil || perLayerQuantization != nil {
        quantize(model: model) { path, module in
            if weights["\(path).scales"] != nil {
                if let perLayerQuantization {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                } else {
                    return quantization?.asTuple
                }
            } else {
                return nil
            }
        }
    }

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])

    eval(model)
}
