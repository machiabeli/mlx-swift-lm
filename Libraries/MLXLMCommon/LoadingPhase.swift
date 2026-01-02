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
