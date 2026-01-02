# LoadingPhase API Implementation Plan

## Overview

This plan adds granular progress tracking for model loading phases to mlx-swift-lm. Currently, the library only reports download progress via `Progress`. This enhancement introduces a `LoadingPhase` enum that provides semantic phase information throughout the entire loading pipeline.

## Current State Analysis

### Existing Progress Handling

The current API uses Foundation's `Progress` object:

```swift
// ModelFactory.swift - Current signature
public func load(
    hub: HubApi = defaultHubApi, configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
) async throws -> sending ModelContext
```

**Limitations:**
1. `Progress` only reports download progress from `hub.snapshot()`
2. No visibility into weight loading, tokenizer loading, or model initialization
3. Large models can take 10-30+ seconds to load weights after download completes, with no feedback

### Loading Pipeline (LLMModelFactory._load)

```
1. downloadModel()           <- Progress reported here
2. JSONDecoder.decode()      <- Silent
3. typeRegistry.createModel() <- Silent
4. loadWeights()             <- Silent (can be 10-30s for large models!)
5. loadTokenizer()           <- Silent (network call possible)
6. Create processor          <- Silent
7. Return ModelContext       <- Done
```

---

## Proposed API

### 1. LoadingPhase Enum

**File:** `Libraries/MLXLMCommon/LoadingPhase.swift` (new file)

```swift
// Copyright © 2024 Apple Inc.

import Foundation

/// Represents the current phase of model loading.
///
/// Use this to provide granular progress feedback during model loading.
/// Each phase represents a distinct step in the loading pipeline.
///
/// Example usage:
/// ```swift
/// let container = try await LLMModelFactory.shared.loadContainer(
///     configuration: config,
///     progressHandler: { progress in
///         // Download progress
///     },
///     phaseHandler: { phase in
///         switch phase {
///         case .downloadingWeights(let file, let index, let total, _):
///             print("Downloading \(file) (\(index)/\(total))")
///         case .loadingWeights(let file, let index, let total):
///             print("Loading weights from \(file) (\(index)/\(total))")
///         case .ready:
///             print("Model ready!")
///         default:
///             break
///         }
///     }
/// )
/// ```
public enum LoadingPhase: Sendable, Equatable {
    /// Downloading model configuration files (config.json, etc.)
    case downloadingConfig

    /// Downloading weight files from the hub.
    /// - Parameters:
    ///   - file: Name of the file being downloaded
    ///   - fileIndex: 1-based index of current file
    ///   - totalFiles: Total number of files to download
    ///   - progress: Download progress for this file (0.0-1.0)
    case downloadingWeights(file: String, fileIndex: Int, totalFiles: Int, progress: Double)

    /// Loading weights from safetensor files into memory.
    /// - Parameters:
    ///   - file: Name of the safetensor file being loaded
    ///   - fileIndex: 1-based index of current file
    ///   - totalFiles: Total number of safetensor files
    case loadingWeights(file: String, fileIndex: Int, totalFiles: Int)

    /// Loading and configuring the tokenizer.
    case loadingTokenizer

    /// Initializing the model with loaded weights.
    /// This includes weight sanitization, quantization application, and model evaluation.
    case initializingModel

    /// Model is fully loaded and ready for inference.
    case ready

    /// Loading failed with an error.
    case failed(LoadingError)

    /// A human-readable description of the current phase.
    public var description: String {
        switch self {
        case .downloadingConfig:
            return "Downloading configuration"
        case .downloadingWeights(let file, let index, let total, let progress):
            let percent = Int(progress * 100)
            return "Downloading \(file) (\(index)/\(total)) - \(percent)%"
        case .loadingWeights(let file, let index, let total):
            return "Loading weights from \(file) (\(index)/\(total))"
        case .loadingTokenizer:
            return "Loading tokenizer"
        case .initializingModel:
            return "Initializing model"
        case .ready:
            return "Ready"
        case .failed(let error):
            return "Failed: \(error.localizedDescription)"
        }
    }

    /// Returns true if this is a terminal phase (ready or failed).
    public var isTerminal: Bool {
        switch self {
        case .ready, .failed:
            return true
        default:
            return false
        }
    }

    /// Estimated progress through the entire loading process (0.0-1.0).
    /// This is approximate and based on typical phase durations.
    public var estimatedProgress: Double {
        switch self {
        case .downloadingConfig:
            return 0.01
        case .downloadingWeights(_, let index, let total, let progress):
            // Downloads typically 0-50% of total time
            let baseProgress = 0.01
            let downloadWeight = 0.49
            let perFileWeight = downloadWeight / Double(max(total, 1))
            return baseProgress + (Double(index - 1) * perFileWeight) + (progress * perFileWeight)
        case .loadingWeights(_, let index, let total):
            // Weight loading typically 50-85% of total time
            let baseProgress = 0.50
            let loadWeight = 0.35
            let perFileWeight = loadWeight / Double(max(total, 1))
            return baseProgress + (Double(index) * perFileWeight)
        case .loadingTokenizer:
            return 0.88
        case .initializingModel:
            return 0.95
        case .ready:
            return 1.0
        case .failed:
            return 0.0
        }
    }
}

/// Errors that can occur during model loading.
public struct LoadingError: Error, Sendable, Equatable, LocalizedError {
    public let phase: String
    public let underlyingError: String

    public init(phase: String, error: Error) {
        self.phase = phase
        self.underlyingError = error.localizedDescription
    }

    public var errorDescription: String? {
        "Loading failed during \(phase): \(underlyingError)"
    }

    public static func == (lhs: LoadingError, rhs: LoadingError) -> Bool {
        lhs.phase == rhs.phase && lhs.underlyingError == rhs.underlyingError
    }
}

/// Type alias for phase handler callbacks.
public typealias PhaseHandler = @Sendable (LoadingPhase) -> Void
```

---

### 2. ModelFactory Protocol Extension

**File:** `Libraries/MLXLMCommon/ModelFactory.swift`

Add new methods to the `ModelFactory` protocol and its extension:

```swift
// Add to ModelFactory protocol (around line 87)
public protocol ModelFactory: Sendable {

    var modelRegistry: AbstractModelRegistry { get }

    func _load(
        hub: HubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> sending ModelContext

    // NEW: Add phase-aware loading
    func _load(
        hub: HubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void,
        phaseHandler: @Sendable @escaping (LoadingPhase) -> Void
    ) async throws -> sending ModelContext

    func _loadContainer(
        hub: HubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> ModelContainer

    // NEW: Add phase-aware container loading
    func _loadContainer(
        hub: HubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void,
        phaseHandler: @Sendable @escaping (LoadingPhase) -> Void
    ) async throws -> ModelContainer
}
```

Add default implementations in the extension:

```swift
extension ModelFactory {
    // ... existing methods ...

    // NEW: Phase-aware load with default empty handler for backward compatibility
    public func load(
        hub: HubApi = defaultHubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in },
        phaseHandler: @Sendable @escaping (LoadingPhase) -> Void = { _ in }
    ) async throws -> sending ModelContext {
        try await _load(
            hub: hub, configuration: configuration,
            progressHandler: progressHandler, phaseHandler: phaseHandler
        )
    }

    // NEW: Phase-aware loadContainer
    public func loadContainer(
        hub: HubApi = defaultHubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in },
        phaseHandler: @Sendable @escaping (LoadingPhase) -> Void = { _ in }
    ) async throws -> ModelContainer {
        try await _loadContainer(
            hub: hub, configuration: configuration,
            progressHandler: progressHandler, phaseHandler: phaseHandler
        )
    }

    // Default implementation that forwards to non-phase version
    public func _load(
        hub: HubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void,
        phaseHandler: @Sendable @escaping (LoadingPhase) -> Void
    ) async throws -> sending ModelContext {
        // Default: just call the non-phase version
        try await _load(hub: hub, configuration: configuration, progressHandler: progressHandler)
    }

    public func _loadContainer(
        hub: HubApi = defaultHubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in },
        phaseHandler: @Sendable @escaping (LoadingPhase) -> Void = { _ in }
    ) async throws -> ModelContainer {
        let context = try await _load(
            hub: hub, configuration: configuration,
            progressHandler: progressHandler, phaseHandler: phaseHandler
        )
        return ModelContainer(context: context)
    }
}
```

---

### 3. Updated Load.swift

**File:** `Libraries/MLXLMCommon/Load.swift`

Add phase-aware versions of `downloadModel` and `loadWeights`:

```swift
/// Download the model with phase reporting.
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

/// Load model weights with phase reporting.
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
```

---

### 4. LLMModelFactory Implementation

**File:** `Libraries/MLXLLM/LLMModelFactory.swift`

Add the phase-aware `_load` implementation:

```swift
public class LLMModelFactory: ModelFactory {
    // ... existing code ...

    // NEW: Phase-aware loading implementation
    public func _load(
        hub: HubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void,
        phaseHandler: @Sendable @escaping (LoadingPhase) -> Void
    ) async throws -> sending ModelContext {
        do {
            // Download weights and config with phase reporting
            let modelDirectory = try await downloadModel(
                hub: hub, configuration: configuration,
                progressHandler: progressHandler,
                phaseHandler: phaseHandler
            )

            // Load the generic config
            let configurationURL = modelDirectory.appending(component: "config.json")

            let baseConfig: BaseConfiguration
            do {
                baseConfig = try JSONDecoder().decode(
                    BaseConfiguration.self, from: Data(contentsOf: configurationURL))
            } catch let error as DecodingError {
                throw ModelFactoryError.configurationDecodingError(
                    configurationURL.lastPathComponent, configuration.name, error)
            }

            let model: LanguageModel
            do {
                model = try typeRegistry.createModel(
                    configuration: configurationURL, modelType: baseConfig.modelType)
            } catch let error as DecodingError {
                throw ModelFactoryError.configurationDecodingError(
                    configurationURL.lastPathComponent, configuration.name, error)
            }

            // Apply weights with phase reporting
            try loadWeights(
                modelDirectory: modelDirectory, model: model,
                perLayerQuantization: baseConfig.perLayerQuantization,
                phaseHandler: phaseHandler
            )

            // Load tokenizer with phase reporting
            phaseHandler(.loadingTokenizer)
            let tokenizer = try await loadTokenizer(configuration: configuration, hub: hub)

            let messageGenerator =
                if let model = model as? LLMModel {
                    model.messageGenerator(tokenizer: tokenizer)
                } else {
                    DefaultMessageGenerator()
                }

            let processor = LLMUserInputProcessor(
                tokenizer: tokenizer, configuration: configuration,
                messageGenerator: messageGenerator)

            phaseHandler(.ready)

            return .init(
                configuration: configuration, model: model, processor: processor, tokenizer: tokenizer)

        } catch {
            phaseHandler(.failed(LoadingError(phase: "loading", error: error)))
            throw error
        }
    }
}
```

---

### 5. VLMModelFactory Implementation

**File:** `Libraries/MLXVLM/VLMModelFactory.swift`

Add the same phase-aware implementation pattern:

```swift
public class VLMModelFactory: ModelFactory {
    // ... existing code ...

    // NEW: Phase-aware loading implementation
    public func _load(
        hub: HubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void,
        phaseHandler: @Sendable @escaping (LoadingPhase) -> Void
    ) async throws -> sending ModelContext {
        do {
            // Download weights and config with phase reporting
            let modelDirectory = try await downloadModel(
                hub: hub, configuration: configuration,
                progressHandler: progressHandler,
                phaseHandler: phaseHandler
            )

            let configurationURL = modelDirectory.appending(component: "config.json")

            let baseConfig: BaseConfiguration
            do {
                baseConfig = try JSONDecoder().decode(
                    BaseConfiguration.self, from: Data(contentsOf: configurationURL))
            } catch let error as DecodingError {
                throw ModelFactoryError.configurationDecodingError(
                    configurationURL.lastPathComponent, configuration.name, error)
            }

            let model: LanguageModel
            do {
                model = try typeRegistry.createModel(
                    configuration: configurationURL, modelType: baseConfig.modelType)
            } catch let error as DecodingError {
                throw ModelFactoryError.configurationDecodingError(
                    configurationURL.lastPathComponent, configuration.name, error)
            }

            // Apply weights with phase reporting
            try loadWeights(
                modelDirectory: modelDirectory, model: model,
                perLayerQuantization: baseConfig.perLayerQuantization,
                phaseHandler: phaseHandler
            )

            // Load tokenizer with phase reporting
            phaseHandler(.loadingTokenizer)
            let tokenizer = try await loadTokenizer(configuration: configuration, hub: hub)

            let processorConfigurationURL = modelDirectory.appending(
                component: "preprocessor_config.json")

            let baseProcessorConfig: BaseProcessorConfiguration
            do {
                baseProcessorConfig = try JSONDecoder().decode(
                    BaseProcessorConfiguration.self,
                    from: Data(contentsOf: processorConfigurationURL))
            } catch let error as DecodingError {
                throw ModelFactoryError.configurationDecodingError(
                    processorConfigurationURL.lastPathComponent, configuration.name, error)
            }

            let processor = try processorRegistry.createModel(
                configuration: processorConfigurationURL,
                processorType: baseProcessorConfig.processorClass, tokenizer: tokenizer)

            phaseHandler(.ready)

            return .init(
                configuration: configuration, model: model, processor: processor, tokenizer: tokenizer)

        } catch {
            phaseHandler(.failed(LoadingError(phase: "loading", error: error)))
            throw error
        }
    }
}
```

---

### 6. Top-Level Convenience Functions

**File:** `Libraries/MLXLMCommon/ModelFactory.swift`

Add phase-aware versions of the top-level functions:

```swift
/// Load a model with phase reporting.
public func loadModel(
    hub: HubApi = defaultHubApi, configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in },
    phaseHandler: @Sendable @escaping (LoadingPhase) -> Void = { _ in }
) async throws -> sending ModelContext {
    try await load {
        try await $0.load(
            hub: hub, configuration: configuration,
            progressHandler: progressHandler, phaseHandler: phaseHandler
        )
    }
}

/// Load a model container with phase reporting.
public func loadModelContainer(
    hub: HubApi = defaultHubApi, configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in },
    phaseHandler: @Sendable @escaping (LoadingPhase) -> Void = { _ in }
) async throws -> sending ModelContainer {
    try await load {
        try await $0.loadContainer(
            hub: hub, configuration: configuration,
            progressHandler: progressHandler, phaseHandler: phaseHandler
        )
    }
}

/// Load a model by ID with phase reporting.
public func loadModel(
    hub: HubApi = defaultHubApi, id: String, revision: String = "main",
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in },
    phaseHandler: @Sendable @escaping (LoadingPhase) -> Void = { _ in }
) async throws -> sending ModelContext {
    try await load {
        try await $0.load(
            hub: hub, configuration: .init(id: id, revision: revision),
            progressHandler: progressHandler, phaseHandler: phaseHandler
        )
    }
}

/// Load a model container by ID with phase reporting.
public func loadModelContainer(
    hub: HubApi = defaultHubApi, id: String, revision: String = "main",
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in },
    phaseHandler: @Sendable @escaping (LoadingPhase) -> Void = { _ in }
) async throws -> sending ModelContainer {
    try await load {
        try await $0.loadContainer(
            hub: hub, configuration: .init(id: id, revision: revision),
            progressHandler: progressHandler, phaseHandler: phaseHandler
        )
    }
}

/// Load a model from directory with phase reporting.
public func loadModel(
    hub: HubApi = defaultHubApi, directory: URL,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in },
    phaseHandler: @Sendable @escaping (LoadingPhase) -> Void = { _ in }
) async throws -> sending ModelContext {
    try await load {
        try await $0.load(
            hub: hub, configuration: .init(directory: directory),
            progressHandler: progressHandler, phaseHandler: phaseHandler
        )
    }
}

/// Load a model container from directory with phase reporting.
public func loadModelContainer(
    hub: HubApi = defaultHubApi, directory: URL,
    progressHandler: @Sendable @escaping (Progress) -> Void = { _ in },
    phaseHandler: @Sendable @escaping (LoadingPhase) -> Void = { _ in }
) async throws -> sending ModelContainer {
    try await load {
        try await $0.loadContainer(
            hub: hub, configuration: .init(directory: directory),
            progressHandler: progressHandler, phaseHandler: phaseHandler
        )
    }
}
```

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `Libraries/MLXLMCommon/LoadingPhase.swift` | **CREATE** | New enum + error types |
| `Libraries/MLXLMCommon/ModelFactory.swift` | MODIFY | Add protocol methods + convenience functions |
| `Libraries/MLXLMCommon/Load.swift` | MODIFY | Add phase-aware download/load functions |
| `Libraries/MLXLLM/LLMModelFactory.swift` | MODIFY | Implement phase-aware `_load` |
| `Libraries/MLXVLM/VLMModelFactory.swift` | MODIFY | Implement phase-aware `_load` |

---

## Backward Compatibility

All changes are **backward compatible**:

1. **Default empty handler:** `phaseHandler: @Sendable @escaping (LoadingPhase) -> Void = { _ in }`
2. **Existing signatures unchanged:** All current method signatures continue to work
3. **Protocol conformance:** Default implementations provided for the new protocol requirements
4. **No breaking changes:** Existing code compiles and runs without modification

---

## Usage Examples

### Basic Usage

```swift
let container = try await loadModelContainer(
    id: "mlx-community/Qwen3-4B-4bit",
    phaseHandler: { phase in
        print(phase.description)
    }
)
```

### SwiftUI Integration

```swift
@Observable
class ModelLoader {
    var phase: LoadingPhase = .downloadingConfig
    var progress: Double = 0
    var container: ModelContainer?

    func load() async throws {
        container = try await loadModelContainer(
            id: "mlx-community/Qwen3-4B-4bit",
            progressHandler: { [weak self] progress in
                Task { @MainActor in
                    self?.progress = progress.fractionCompleted
                }
            },
            phaseHandler: { [weak self] phase in
                Task { @MainActor in
                    self?.phase = phase
                }
            }
        )
    }
}

struct LoadingView: View {
    @State private var loader = ModelLoader()

    var body: some View {
        VStack {
            ProgressView(value: loader.phase.estimatedProgress)
            Text(loader.phase.description)
                .font(.caption)
        }
    }
}
```

### Error Handling

```swift
do {
    let container = try await loadModelContainer(
        id: "invalid-model",
        phaseHandler: { phase in
            if case .failed(let error) = phase {
                logger.error("Loading failed in \(error.phase): \(error.underlyingError)")
            }
        }
    )
} catch {
    // Handle error
}
```

---

## Testing Strategy

1. **Unit tests for LoadingPhase:**
   - Test `description` for all cases
   - Test `isTerminal` property
   - Test `estimatedProgress` ranges
   - Test `Equatable` conformance

2. **Integration tests:**
   - Load a small model and verify phase sequence
   - Verify phases emitted in correct order
   - Test with cached model (skip download phases)
   - Test with invalid model (verify `.failed` phase)

3. **Thread safety:**
   - Verify `@Sendable` compliance
   - Test concurrent loads with different phase handlers

---

## Future Enhancements

1. **Per-file download progress:** Requires Hub API changes to expose file-level progress
2. **Memory usage reporting:** Add memory estimates per phase
3. **Cancellation support:** Allow cancelling load at any phase
4. **Retry support:** Retry failed phases automatically
5. **AsyncSequence API:** `for await phase in loadModelPhases(...)`
