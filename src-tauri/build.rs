fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if target_os == "macos" {
        // Add Swift runtime rpath for ScreenCaptureKit dependency
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    }

    if target_os == "windows" {
        // Required by onnxruntime static linking (ETW telemetry + registry)
        println!("cargo:rustc-link-lib=advapi32");
        println!("cargo:rustc-link-lib=advevtapi");
    }

    tauri_build::build()
}
