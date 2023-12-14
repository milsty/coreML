//
//  BananaCoreML.swift
//  CoreMLDemo
//
//  Created by kuraki on 2023/12/7.
//  Copyright © 2023 Weibo. All rights reserved.
//

import Foundation
import CoreImage
import CreateMLComponents
import SwiftUI

@objcMembers public class BananaModel : NSObject {
    var regressor :ImageRegressor?
    var transformer : Any?
    
    func trainBananaRipe(url:URL) async -> String {
        do {
            self.regressor = ImageRegressor()
            self.transformer = try await self.regressor?.train(trainingDataURL: url)
            return self.regressor?.trainMessage ?? ""
        } catch {
            print("train error")
            return ""
        }
    }
    
    func predictBananaRipe(image:CIImage) async -> Float {
        guard let regressor = self.regressor else {
            return -100;
        }
        if let transformer = self.transformer as? any Transformer<CIImage, Float> {
            do {
                let result = try await transformer.applied(to: image, eventHandler: {_ in })
                return result
            } catch {
                print("test error")
                return -100;
            }
        }
        return -100;
    }
}

@objcMembers public class ImageRegressor : NSObject {
    static let parametersURL = URL(fileURLWithPath: "~/parameters")
    var trainMessage : String = ""
    func train(trainingDataURL:URL) async throws -> some Transformer<CIImage, Float> {
        let estimator = ImageFeaturePrint()
            .appending(LinearRegressor())
        
        let data = try AnnotatedFiles(labeledByNamesAt: trainingDataURL, separator: "-", index: 1, type: .image)
            .mapFeatures(ImageReader.read)
            .mapAnnotations({ Float($0)! })
            .flatMap(augment)

        let (training, validation) = data.randomSplit(by: 0.8)
        let transformer = try await estimator.fitted(to: training, validateOn: validation) { [weak self] event in
            guard let trainingMaxError = event.metrics[.trainingMaximumError] else {
                return
            }
            guard let validationMaxError = event.metrics[.validationMaximumError] else {
                return
            }
            self?.trainMessage = self?.trainMessage.appending("\n") ?? ""
            self?.trainMessage = self?.trainMessage.appending("Training max error: \(trainingMaxError), Validation max error: \(validationMaxError)") ?? ""
            print("Training max error: \(trainingMaxError), Validation max error: \(validationMaxError)")
        }

        let validationError = try await meanAbsoluteError(
            transformer.applied(to: validation.map(\.feature)),
            validation.map(\.annotation)
        )
        self.trainMessage = self.trainMessage.appending("\n") 
        self.trainMessage = self.trainMessage.appending("Mean absolute error: \(validationError)")
        print("Mean absolute error: \(validationError)")

        try estimator.write(transformer, to: ImageRegressor.parametersURL)
        return transformer
    }

    func augment(_ original: AnnotatedFeature<CIImage, Float>) -> [AnnotatedFeature<CIImage, Float>] {
        let angle = CGFloat.random(in: -.pi ... .pi)
        let rotated = original.feature.transformed(by: .init(rotationAngle: angle))

        let scale = CGFloat.random(in: 0.8 ... 1.2)
        let scaled = original.feature.transformed(by: .init(scaleX: scale, y: scale))

        return [
            original,
            AnnotatedFeature(feature: rotated, annotation: original.annotation),
            AnnotatedFeature(feature: scaled, annotation: original.annotation),
        ]
    }
}
