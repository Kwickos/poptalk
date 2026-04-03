pub mod audio;
pub mod whisper;
pub mod diarization;
pub mod models;
pub mod llm;
pub mod db;
pub mod export;
pub mod commands;
pub mod app_state;

use app_state::AppState;
use db::Database;
use tauri::{
    menu::{AboutMetadataBuilder, MenuBuilder, MenuItemBuilder, PredefinedMenuItem, SubmenuBuilder},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    Emitter, Manager, WindowEvent,
};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let db_path = dirs::data_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("poptalk")
        .join("poptalk.db");

    // Create parent directory if needed
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let db = Database::new(&db_path).expect("Failed to open database");

    // Load API key from settings
    let api_key = db.get_setting("api_key").ok().flatten().unwrap_or_default();
    let state = AppState::new(db);
    *state.api_key.lock().unwrap() = api_key;

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(state)
        .invoke_handler(tauri::generate_handler![
            commands::start_session,
            commands::stop_session,
            commands::get_sessions,
            commands::get_session_detail,
            commands::search_text,
            commands::search_llm,
            commands::rename_speaker,
            commands::export_session,
            commands::update_session_title,
            commands::delete_session,
            commands::get_api_key,
            commands::set_api_key,
            commands::get_setting,
            commands::set_setting,
            commands::list_input_devices,
            commands::pick_folder,
            commands::get_available_models,
            commands::get_model_status,
            commands::download_model,
        ])
        .setup(|app| {
            // --- Application menu bar ---
            let about = PredefinedMenuItem::about(app, Some("A propos de PopTalk"), Some(
                AboutMetadataBuilder::new()
                    .name(Some("PopTalk"))
                    .version(Some(env!("CARGO_PKG_VERSION")))
                    .build()
            ))?;
            let settings_item = MenuItemBuilder::with_id("settings", "Parametres...")
                .accelerator("CmdOrCtrl+,")
                .build(app)?;
            let separator = PredefinedMenuItem::separator(app)?;
            let quit_menu = PredefinedMenuItem::quit(app, Some("Quitter PopTalk"))?;

            #[cfg(target_os = "macos")]
            let app_submenu = {
                let hide = PredefinedMenuItem::hide(app, Some("Masquer PopTalk"))?;
                let hide_others = PredefinedMenuItem::hide_others(app, Some("Masquer les autres"))?;
                let show_all = PredefinedMenuItem::show_all(app, Some("Tout afficher"))?;

                SubmenuBuilder::new(app, "PopTalk")
                    .item(&about)
                    .separator()
                    .item(&settings_item)
                    .item(&separator)
                    .item(&hide)
                    .item(&hide_others)
                    .item(&show_all)
                    .separator()
                    .item(&quit_menu)
                    .build()?
            };

            #[cfg(not(target_os = "macos"))]
            let app_submenu = SubmenuBuilder::new(app, "PopTalk")
                .item(&about)
                .separator()
                .item(&settings_item)
                .item(&separator)
                .item(&quit_menu)
                .build()?;

            #[cfg(target_os = "macos")]
            let edit_submenu = SubmenuBuilder::new(app, "Edition")
                .undo()
                .redo()
                .separator()
                .cut()
                .copy()
                .paste()
                .select_all()
                .build()?;

            #[cfg(not(target_os = "macos"))]
            let edit_submenu = SubmenuBuilder::new(app, "Edition")
                .cut()
                .copy()
                .paste()
                .select_all()
                .build()?;

            let app_menu = MenuBuilder::new(app)
                .item(&app_submenu)
                .item(&edit_submenu)
                .build()?;

            app.set_menu(app_menu)?;

            // Handle menu events (settings)
            app.on_menu_event(move |app_handle, event| {
                if event.id().as_ref() == "settings" {
                    let _ = app_handle.emit("open-settings", ());
                }
            });

            // --- System tray ---
            let open_item = MenuItemBuilder::with_id("open", "Ouvrir PopTalk").build(app)?;
            let quit_item = MenuItemBuilder::with_id("quit", "Quitter").build(app)?;
            let tray_menu = MenuBuilder::new(app)
                .items(&[&open_item])
                .separator()
                .items(&[&quit_item])
                .build()?;

            // Create the system tray icon
            let _tray = TrayIconBuilder::new()
                .icon(app.default_window_icon().unwrap().clone())
                .menu(&tray_menu)
                .on_menu_event(|app, event| match event.id().as_ref() {
                    "open" => {
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.unminimize();
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                    "quit" => {
                        app.exit(0);
                    }
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        let app = tray.app_handle();
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.unminimize();
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                })
                .build(app)?;

            Ok(())
        })
        .on_window_event(|window, event| {
            // Intercept window close: hide instead of quitting so the app stays in the tray
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.hide();
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
