import Foundation
import Security

enum KeychainStore {
    private static let service = "com.drewboynton.llmadvisor"

    @discardableResult
    static func save(_ value: String, key: String) -> OSStatus {
        #if targetEnvironment(simulator)
        // Unsigned simulator builds can lack the keychain entitlement. Keep
        // this fallback simulator-only; physical Debug builds still use the
        // device keychain.
        UserDefaults.standard.set(value, forKey: "llm-advisor.simulator-keychain." + key)
        return errSecSuccess
        #endif
        guard let data = value.data(using: .utf8) else { return errSecParam }
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
        ]
        SecItemDelete(query as CFDictionary)
        var item = query
        item[kSecValueData as String] = data
        item[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        let status = SecItemAdd(item as CFDictionary, nil)
        return status
    }

    static func read(key: String) -> String? {
        #if targetEnvironment(simulator)
        return UserDefaults.standard.string(forKey: "llm-advisor.simulator-keychain." + key)
        #endif
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess,
              let data = result as? Data else {
            return nil
        }
        return String(data: data, encoding: .utf8)
    }

    static func delete(key: String) {
        #if targetEnvironment(simulator)
        UserDefaults.standard.removeObject(forKey: "llm-advisor.simulator-keychain." + key)
        return
        #endif
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
        ]
        SecItemDelete(query as CFDictionary)
    }
}
