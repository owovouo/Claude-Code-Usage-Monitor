#![windows_subsystem = "windows"]

mod diagnose;
mod localization;
mod models;
mod native_interop;
mod poller;
mod theme;
mod tray_icon;
mod updater;
mod window;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // TEMP (2026-06-14): force diagnose on while investigating the Claude vs
    // Codex reset-time / quiet-hours mismatch, so both machines log without
    // needing the flag. Revert to
    // `args.iter().any(|arg| arg == "--diagnose")` once resolved.
    let diagnose_enabled = true;
    if diagnose_enabled {
        match diagnose::init() {
            Ok(path) => diagnose::log(format!(
                "startup args={args:?} log_path={}",
                path.display()
            )),
            Err(error) => {
                // Logging may not be available yet, but keep startup behavior unchanged.
                let _ = error;
            }
        }
    }

    if args.iter().any(|arg| arg == "--force-codex-trigger") {
        poller::arm_force_codex_trigger();
        diagnose::log("--force-codex-trigger armed: next Codex poll will bypass all gates");
    }

    if let Some(exit_code) = updater::handle_cli_mode(&args) {
        if diagnose_enabled {
            diagnose::log(format!("cli mode exited with code {exit_code}"));
        }
        std::process::exit(exit_code);
    }

    if diagnose_enabled {
        diagnose::log("entering window::run");
    }
    window::run();
}
