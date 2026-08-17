import UIKit
import SwiftUI

class SceneDelegate: UIResponder, UIWindowSceneDelegate {
    var window: UIWindow?

    func scene(_ scene: UIScene, willConnectTo session: UISceneSession, options connectionOptions: UIScene.ConnectionOptions) {
        guard let windowScene = scene as? UIWindowScene else { return }

        window = UIWindow(windowScene: windowScene)
        window?.rootViewController = NativeHostingController(rootView: NativeAppView())
        window?.makeKeyAndVisible()

        connectionOptions.urlContexts.forEach { context in
            NotificationCenter.default.post(name: .advisorDeepLink, object: context.url)
        }
    }

    func scene(_ scene: UIScene, openURLContexts URLContexts: Set<UIOpenURLContext>) {
        URLContexts.forEach { context in
            NotificationCenter.default.post(name: .advisorDeepLink, object: context.url)
        }
    }

    func scene(_ scene: UIScene, continue userActivity: NSUserActivity) {
        if let url = userActivity.webpageURL {
            NotificationCenter.default.post(name: .advisorDeepLink, object: url)
        }
    }
}

final class NativeHostingController: UIHostingController<NativeAppView> {
    override var preferredStatusBarStyle: UIStatusBarStyle {
        .darkContent
    }
}
