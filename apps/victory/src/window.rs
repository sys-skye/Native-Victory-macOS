//! The window of the native macOS host: an `NSWindow` (1920x1080 content, "VIctory") whose view
//! is the `CAMetalLayer` the GPU presents into, driven by an AppKit event loop on the process
//! main thread while the title runs on its own threads. Closing the window ends the process.
//!
//! AppKit is reached through the Objective-C runtime (`objc2`'s `msg_send!`), with no binding
//! crate for it. The loop is `nextEventMatchingMask:` by hand rather than `[NSApp run]`, so the
//! title finishing is noticed without a delegate class.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// No binding crate links AppKit or GameController, so their classes would not be registered
// without this.
#[link(name = "AppKit", kind = "framework")]
#[link(name = "GameController", kind = "framework")]
unsafe extern "C" {}

use anyhow::{Context, Result};
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{AnyObject, Bool};
use objc2::{class, msg_send};
use objc2_foundation::{NSPoint, NSRect, NSSize, NSString};

/// Content size in points.
pub const WIDTH: usize = 1920;
pub const HEIGHT: usize = 1080;

/// Shows a window over `layer` (a `CAMetalLayer`) and runs `title` on a thread of its own until
/// it returns, or the window is closed (then the process exits). Call on the main thread.
pub fn run<T: Send + 'static>(
    layer: *mut std::ffi::c_void,
    title: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<Option<T>> {
    anyhow::ensure!(
        // SAFETY: a plain libc query.
        unsafe { pthread_main_np() } != 0,
        "the window needs the process main thread"
    );
    // SAFETY: AppKit calls on the main thread with objects of the right classes.
    let window = unsafe { open(layer) }?;
    let done: Arc<Mutex<Option<Result<T>>>> = Arc::default();
    let slot = done.clone();
    std::thread::Builder::new()
        .name("title-launch".into())
        .spawn(move || {
            let result = title();
            *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(result);
        })
        .context("starting the title")?;
    // SAFETY: as above.
    unsafe { event_loop(&window, || done.lock().is_ok_and(|d| d.is_some())) };
    let finished = done.lock().unwrap_or_else(|e| e.into_inner()).take();
    match finished {
        Some(result) => result.map(Some),
        None => {
            // The window was closed.
            use std::io::Write;
            let _ = std::io::stdout().flush();
            std::process::exit(0);
        }
    }
}

unsafe extern "C" {
    fn pthread_main_np() -> i32;
}

/// The application, its menu (Quit) and the window, shown.
unsafe fn open(layer: *mut std::ffi::c_void) -> Result<Retained<AnyObject>> {
    // SAFETY: documented AppKit calls; every receiver is a class or an object just made.
    unsafe {
        let app: *mut AnyObject = msg_send![class!(NSApplication), sharedApplication];
        anyhow::ensure!(!app.is_null(), "no NSApplication");
        let _: Bool = msg_send![app, setActivationPolicy: 0isize]; // regular: a Dock icon and menu
        let menubar: Retained<AnyObject> = msg_send![class!(NSMenu), new];
        let item: Retained<AnyObject> = msg_send![class!(NSMenuItem), new];
        let _: () = msg_send![&*menubar, addItem: &*item];
        let menu: Retained<AnyObject> = msg_send![class!(NSMenu), new];
        let quit: Retained<AnyObject> = msg_send![
            msg_send![class!(NSMenuItem), alloc],
            initWithTitle: &*NSString::from_str("Quit VIctory"),
            action: objc2::sel!(terminate:),
            keyEquivalent: &*NSString::from_str("q")
        ];
        let _: () = msg_send![&*menu, addItem: &*quit];
        let _: () = msg_send![&*item, setSubmenu: &*menu];
        let _: () = msg_send![app, setMainMenu: &*menubar];

        let rect = NSRect::new(
            NSPoint::new(0.0, 0.0),
            NSSize::new(WIDTH as f64, HEIGHT as f64),
        );
        // titled | closable | miniaturizable | resizable; buffered.
        let window: Retained<AnyObject> = msg_send![
            msg_send![class!(NSWindow), alloc],
            initWithContentRect: rect,
            styleMask: 15usize,
            backing: 2usize,
            defer: Bool::NO
        ];
        let _: () = msg_send![&*window, setTitle: &*NSString::from_str("VIctory")];
        // The view hosts our layer: set the layer first, then ask for layers.
        let view: *mut AnyObject = msg_send![&*window, contentView];
        let layer = layer.cast::<AnyObject>();
        let _: () = msg_send![layer, setFrame: rect];
        let _: () = msg_send![layer, setAutoresizingMask: 18u32]; // width and height sizable
        let _: () = msg_send![view, setLayer: layer];
        let _: () = msg_send![view, setWantsLayer: Bool::YES];
        let _: () = msg_send![&*window, center];
        let _: () = msg_send![&*window, makeKeyAndOrderFront: std::ptr::null::<AnyObject>()];
        let _: () = msg_send![app, finishLaunching];
        let _: () = msg_send![app, activateIgnoringOtherApps: Bool::YES];
        Ok(window)
    }
}

/// macOS Carbon virtual key code (`kVK_*`) to Windows virtual key (`VK_*`).
fn virtual_key(code: u16) -> Option<u8> {
    Some(match code {
        0 => b'A',
        1 => b'S',
        2 => b'D',
        3 => b'F',
        4 => b'H',
        5 => b'G',
        6 => b'Z',
        7 => b'X',
        8 => b'C',
        9 => b'V',
        11 => b'B',
        12 => b'Q',
        13 => b'W',
        14 => b'E',
        15 => b'R',
        16 => b'Y',
        17 => b'T',
        18 => b'1',
        19 => b'2',
        20 => b'3',
        21 => b'4',
        22 => b'6',
        23 => b'5',
        24 => 0xBB, // =
        25 => b'9',
        26 => b'7',
        27 => 0xBD, // -
        28 => b'8',
        29 => b'0',
        30 => 0xDD, // ]
        31 => b'O',
        32 => b'U',
        33 => 0xDB, // [
        34 => b'I',
        35 => b'P',
        36 | 76 => 0x0D, // Return, keypad Enter
        37 => b'L',
        38 => b'J',
        39 => 0xDE, // '
        40 => b'K',
        41 => 0xBA, // ;
        42 => 0xDC, // \
        43 => 0xBC, // ,
        44 => 0xBF, // /
        45 => b'N',
        46 => b'M',
        47 => 0xBE, // .
        48 => 0x09, // Tab
        49 => 0x20, // Space
        51 => 0x08, // Delete (Backspace)
        53 => 0x1B, // Escape
        96 => 0x74, // F5
        97 => 0x75, // F6
        98 => 0x76, // F7
        99 => 0x72, // F3
        100 => 0x77, // F8
        101 => 0x78, // F9
        103 => 0x7A, // F11
        109 => 0x79, // F10
        111 => 0x7B, // F12
        115 => 0x24, // Home
        116 => 0x21, // Page Up
        117 => 0x2E, // Forward Delete
        118 => 0x73, // F4
        119 => 0x23, // End
        120 => 0x71, // F2
        121 => 0x22, // Page Down
        122 => 0x70, // F1
        123 => 0x25, // Left Arrow
        124 => 0x27, // Right Arrow
        125 => 0x28, // Down Arrow
        126 => 0x26, // Up Arrow
        _ => return None,
    })
}

/// Maps a macOS Carbon modifier key code (`kVK_*`) to its Windows virtual key
/// and corresponding `NSEventModifierFlags` bit.
fn modifier_key(code: u16) -> Option<(u8, usize)> {
    match code {
        0x38 => Some((0xA0, 1 << 17)), // Left Shift
        0x3C => Some((0xA1, 1 << 17)), // Right Shift
        0x3B => Some((0xA2, 1 << 18)), // Left Control
        0x3E => Some((0xA3, 1 << 18)), // Right Control
        0x3A => Some((0xA4, 1 << 19)), // Left Option -> Left Alt
        0x3D => Some((0xA5, 1 << 19)), // Right Option -> Right Alt
        _ => None,
    }
}

/// Hands a key-down, key-up or modifier flags-changed event to `GameInput`.
/// Returns `true` when it was a gameplay key-down/key-up (Command shortcuts
/// such as Cmd+Q and flags-changed events are left to AppKit), so the caller
/// does not forward it and AppKit does not beep.
unsafe fn key_event(event: *mut AnyObject) -> bool {
    // SAFETY: `event` is a live NSEvent; `type` is valid for every event.
    let kind: usize = unsafe { msg_send![event, type] };
    if kind == 12 {
        // NSEventTypeFlagsChanged: modifier key pressed or released.
        // SAFETY: `keyCode` and `modifierFlags` are valid on flags-changed events.
        let (code, flags): (u16, usize) =
            unsafe { (msg_send![event, keyCode], msg_send![event, modifierFlags]) };
        if let Some((vk, mask)) = modifier_key(code) {
            victory_hle::modules::gameinput::set_live_key(vk, flags & mask != 0);
        }
        return false;
    }
    if !matches!(kind, 10 | 11) {
        return false;
    }
    // Only key events have a key code: NSEvent raises for any other kind.
    // SAFETY: a key-down or key-up NSEvent; these are its plain getters.
    let (code, flags): (u16, usize) =
        unsafe { (msg_send![event, keyCode], msg_send![event, modifierFlags]) };
    if flags & (1 << 20) != 0 {
        return false;
    }
    if let Some(vk) = virtual_key(code) {
        victory_hle::modules::gameinput::set_live_key(vk, kind == 10);
    }
    true
}

/// The controller `poll_controller` read last (its address, 0 for none), so
/// that a change of controller is logged once.
static ACTIVE: AtomicUsize = AtomicUsize::new(0);

/// The `GCController` that drives the pad: the one used last
/// (`GCController.current`) when it has an extended gamepad, otherwise the
/// first connected one that has. Null when none is connected.
unsafe fn active_controller() -> *mut AnyObject {
    // SAFETY: documented GameController class methods and getters.
    unsafe {
        let current: *mut AnyObject = msg_send![class!(GCController), current];
        if !current.is_null() {
            let gp: *mut AnyObject = msg_send![current, extendedGamepad];
            if !gp.is_null() {
                return current;
            }
        }
        let controllers: *mut AnyObject = msg_send![class!(GCController), controllers];
        if controllers.is_null() {
            return std::ptr::null_mut();
        }
        let count: usize = msg_send![controllers, count];
        for i in 0..count {
            let ctrl: *mut AnyObject = msg_send![controllers, objectAtIndex: i];
            if ctrl.is_null() {
                continue;
            }
            let gp: *mut AnyObject = msg_send![ctrl, extendedGamepad];
            if !gp.is_null() {
                return ctrl;
            }
        }
        std::ptr::null_mut()
    }
}

/// Whether `gamepad` is a PlayStation profile (`GCDualShockGamepad` or
/// `GCDualSenseGamepad`), which add a touchpad to the extended gamepad.
unsafe fn has_touchpad(gamepad: *mut AnyObject) -> bool {
    // SAFETY: `respondsToSelector:` is valid on every NSObject.
    let responds: Bool =
        unsafe { msg_send![gamepad, respondsToSelector: objc2::sel!(touchpadButton)] };
    responds.as_bool()
}

/// Reads a `GCExtendedGamepad` (Xbox, DualShock 4, DualSense or MFi) into a
/// snapshot in the Xbox layout. On a DualShock 4, Cross/Circle/Square/Triangle
/// are A/B/X/Y, Options is Menu, and Share and the touchpad click are View.
unsafe fn snapshot_of(gp: *mut AnyObject) -> victory_hle::modules::gameinput::ControllerSnapshot {
    use victory_hle::modules::gameinput::ControllerSnapshot;
    // SAFETY: documented GameController getters on a live extended gamepad;
    // every element is checked for null before use.
    unsafe {
        let pressed = |btn: *mut AnyObject| -> bool {
            !btn.is_null() && {
                let b: Bool = msg_send![btn, isPressed];
                b.as_bool()
            }
        };
        let trigger = |btn: *mut AnyObject| -> f32 {
            if btn.is_null() { 0.0 } else { msg_send![btn, value] }
        };
        let axis = |dpad: *mut AnyObject| -> (f32, f32) {
            if dpad.is_null() {
                return (0.0, 0.0);
            }
            let x_axis: *mut AnyObject = msg_send![dpad, xAxis];
            let y_axis: *mut AnyObject = msg_send![dpad, yAxis];
            let x: f32 = if x_axis.is_null() { 0.0 } else { msg_send![x_axis, value] };
            let y: f32 = if y_axis.is_null() { 0.0 } else { msg_send![y_axis, value] };
            (x, y)
        };
        let dpad: *mut AnyObject = msg_send![gp, dpad];
        let (up, down, left, right) = if dpad.is_null() {
            (false, false, false, false)
        } else {
            (
                pressed(msg_send![dpad, up]),
                pressed(msg_send![dpad, down]),
                pressed(msg_send![dpad, left]),
                pressed(msg_send![dpad, right]),
            )
        };
        let (left_x, left_y) = axis(msg_send![gp, leftThumbstick]);
        let (right_x, right_y) = axis(msg_send![gp, rightThumbstick]);
        let touchpad = has_touchpad(gp) && pressed(msg_send![gp, touchpadButton]);
        ControllerSnapshot {
            a: pressed(msg_send![gp, buttonA]),
            b: pressed(msg_send![gp, buttonB]),
            x: pressed(msg_send![gp, buttonX]),
            y: pressed(msg_send![gp, buttonY]),
            up,
            down,
            left,
            right,
            left_shoulder: pressed(msg_send![gp, leftShoulder]),
            right_shoulder: pressed(msg_send![gp, rightShoulder]),
            left_stick: pressed(msg_send![gp, leftThumbstickButton]),
            right_stick: pressed(msg_send![gp, rightThumbstickButton]),
            menu: pressed(msg_send![gp, buttonMenu]),
            view: pressed(msg_send![gp, buttonOptions]),
            touchpad,
            left_trigger: trigger(msg_send![gp, leftTrigger]),
            right_trigger: trigger(msg_send![gp, rightTrigger]),
            left_x,
            left_y,
            right_x,
            right_y,
        }
    }
}

/// `s` (an `NSString`, or null) as text.
unsafe fn text(s: *mut AnyObject) -> String {
    if s.is_null() {
        return "unknown".into();
    }
    // SAFETY: the caller passes an NSString.
    unsafe { (*s.cast::<NSString>()).to_string() }
}

/// Logs, once per change, which controller now drives the pad, so a tester
/// can see in the run log that their controller was recognised.
unsafe fn note_controller(ctrl: *mut AnyObject) {
    let id = ctrl as usize;
    if ACTIVE.swap(id, Ordering::Relaxed) == id {
        return;
    }
    if ctrl.is_null() {
        tracing::info!("controller: none connected, the keyboard drives the pad");
        return;
    }
    // SAFETY: documented GCController getters on a live controller.
    unsafe {
        let gp: *mut AnyObject = msg_send![ctrl, extendedGamepad];
        let vendor = text(msg_send![ctrl, vendorName]);
        let category = text(msg_send![ctrl, productCategory]);
        let touchpad = !gp.is_null() && has_touchpad(gp);
        tracing::info!(vendor, category, touchpad, "controller: drives the pad");
    }
}

/// Polls the active `GCController` (see `active_controller`) and updates
/// `GameInput`. Releases controller state when no extended gamepad is
/// connected.
unsafe fn poll_controller() {
    // SAFETY: GameController queries on the main thread.
    unsafe {
        let ctrl = active_controller();
        note_controller(ctrl);
        if ctrl.is_null() {
            victory_hle::modules::gameinput::release_live_pad();
            return;
        }
        let gp: *mut AnyObject = msg_send![ctrl, extendedGamepad];
        victory_hle::modules::gameinput::set_live_pad(snapshot_of(gp));
    }
}

/// Runs the loop until `finished`, or the window is gone.
unsafe fn event_loop(window: &AnyObject, finished: impl Fn() -> bool) {
    let mode = NSString::from_str("kCFRunLoopDefaultMode"); // NSDefaultRunLoopMode
    loop {
        let go_on = autoreleasepool(|_| {
            // SAFETY: AppKit calls on the main thread.
            unsafe {
                let app: *mut AnyObject = msg_send![class!(NSApplication), sharedApplication];
                let until: *mut AnyObject =
                    msg_send![class!(NSDate), dateWithTimeIntervalSinceNow: 0.01f64];
                let event: *mut AnyObject = msg_send![
                    app,
                    nextEventMatchingMask: u64::MAX,
                    untilDate: until,
                    inMode: &*mode,
                    dequeue: Bool::YES
                ];
                if !event.is_null() && !key_event(event) {
                    let _: () = msg_send![app, sendEvent: event];
                }
                let key: Bool = msg_send![window, isKeyWindow];
                if !key.as_bool() {
                    // A key-up or release sent to another window never reaches us.
                    victory_hle::modules::gameinput::release_live_keys();
                    victory_hle::modules::gameinput::release_live_pad();
                } else {
                    poll_controller();
                }
                let _: () = msg_send![app, updateWindows];
                let shown: Bool = msg_send![window, isVisible];
                let minimized: Bool = msg_send![window, isMiniaturized];
                // NSWindowOcclusionStateVisible (1 << 1): some of the window can be seen.
                let occlusion: usize = msg_send![window, occlusionState];
                victory_gpu::set_window_hidden(minimized.as_bool() || occlusion & 2 == 0);
                (shown.as_bool() || minimized.as_bool()) && !finished()
            }
        });
        if !go_on {
            victory_hle::modules::gameinput::release_live_keys();
            victory_hle::modules::gameinput::release_live_pad();
            // SAFETY: as above.
            let shown: Bool = unsafe { msg_send![window, isVisible] };
            if shown.as_bool() {
                // The title ended with the window still up: close it.
                // SAFETY: as above.
                unsafe {
                    let _: () = msg_send![window, close];
                }
            }
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_map_to_the_pad_keys() {
        // Return, Space, Esc, Delete, Tab, Left; W (13), A (0), D (2), Q (12), E (14), Z (6), C (8).
        let got = [36, 49, 53, 51, 48, 123, 13, 0, 2, 12, 14, 6, 8].map(virtual_key);
        let want = [0x0D, 0x20, 0x1B, 0x08, 0x09, 0x25, b'W', b'A', b'D', b'Q', b'E', b'Z', b'C'];
        assert_eq!(got, want.map(Some));
        assert_eq!(virtual_key(10), None); // section sign: not mapped
        assert_eq!(virtual_key(50), None);
        // Full keyboard additions: digits, function keys, navigation, modifiers.
        assert_eq!(virtual_key(18), Some(b'1'));
        assert_eq!(virtual_key(122), Some(0x70)); // F1
        assert_eq!(virtual_key(117), Some(0x2E)); // Forward Delete
        assert_eq!(modifier_key(0x38), Some((0xA0, 1 << 17)));
    }

    /// Exercises `key_event` with real AppKit `NSEvent` objects: key-down,
    /// key-up, Cmd+Q forwarding, `FlagsChanged` modifiers, and non-key events
    /// (verifying `keyCode` is never called on non-key events).
    #[test]
    fn real_nsevents_update_live_keys_and_ignore_non_key_events() {
        autoreleasepool(|_| {
            victory_hle::modules::gameinput::release_live_keys();
            let make_key = |kind: usize, flags: usize, code: u16| -> *mut AnyObject {
                let empty = NSString::from_str("");
                let null: *mut AnyObject = std::ptr::null_mut();
                // SAFETY: documented AppKit factory for synthetic key NSEvents.
                unsafe {
                    msg_send![
                        class!(NSEvent),
                        keyEventWithType: kind,
                        location: NSPoint::new(0.0, 0.0),
                        modifierFlags: flags,
                        timestamp: 0.0f64,
                        windowNumber: 0isize,
                        context: null,
                        characters: &*empty,
                        charactersIgnoringModifiers: &*empty,
                        isARepeat: Bool::NO,
                        keyCode: code
                    ]
                }
            };

            // Gameplay key-down (Return = 36 -> 0x0D) is consumed (returns true).
            let ret_down = make_key(10, 0, 36);
            assert!(!ret_down.is_null());
            // SAFETY: real NSEvent just constructed.
            assert!(unsafe { key_event(ret_down) });
            assert!(victory_hle::modules::gameinput::is_live_key_down(0x0D));

            // Key-up releases it.
            let ret_up = make_key(11, 0, 36);
            // SAFETY: real NSEvent just constructed.
            assert!(unsafe { key_event(ret_up) });
            assert!(!victory_hle::modules::gameinput::is_live_key_down(0x0D));

            // Cmd+Q (code 12, Command = 1 << 20) is left to AppKit (returns false).
            let cmd_q = make_key(10, 1 << 20, 12);
            // SAFETY: real NSEvent just constructed.
            assert!(!unsafe { key_event(cmd_q) });
            assert!(!victory_hle::modules::gameinput::is_live_key_down(b'Q'));

            // FlagsChanged for Left Shift (0x38 -> 0xA0) down then up.
            let shift_down = make_key(12, 1 << 17, 0x38);
            // SAFETY: real NSEvent just constructed.
            assert!(!unsafe { key_event(shift_down) });
            assert!(victory_hle::modules::gameinput::is_live_key_down(0xA0));
            let shift_up = make_key(12, 0, 0x38);
            // SAFETY: real NSEvent just constructed.
            assert!(!unsafe { key_event(shift_up) });
            assert!(!victory_hle::modules::gameinput::is_live_key_down(0xA0));

            // Non-key event (NSApplicationDefined = 15) must not raise on keyCode.
            // SAFETY: documented AppKit factory for synthetic other NSEvents.
            let other: *mut AnyObject = unsafe {
                msg_send![
                    class!(NSEvent),
                    otherEventWithType: 15usize,
                    location: NSPoint::new(0.0, 0.0),
                    modifierFlags: 0usize,
                    timestamp: 0.0f64,
                    windowNumber: 0isize,
                    context: std::ptr::null_mut::<AnyObject>(),
                    subtype: 0i16,
                    data1: 0isize,
                    data2: 0isize
                ]
            };
            assert!(!other.is_null());
            // SAFETY: real NSEvent just constructed.
            assert!(!unsafe { key_event(other) });

            victory_hle::modules::gameinput::release_live_keys();
        });
    }

    /// Sets a button of a software gamepad (`controllerWithExtendedGamepad`).
    unsafe fn press(button: *mut AnyObject, value: f32) {
        assert!(!button.is_null());
        // SAFETY: `setValue:` is documented for snapshot controllers' elements.
        unsafe {
            let _: () = msg_send![button, setValue: value];
        }
    }

    /// Sets a stick or D-pad of a software gamepad.
    unsafe fn push(pad: *mut AnyObject, x: f32, y: f32) {
        assert!(!pad.is_null());
        // SAFETY: as in `press`.
        unsafe {
            let _: () = msg_send![pad, setValueForXAxis: x, yAxis: y];
        }
    }

    /// Reads a software extended gamepad, made without any hardware, through
    /// the same path a real Xbox pad or DualShock 4 takes. It has no
    /// touchpad, so `touchpad` stays up.
    #[test]
    fn snapshot_reads_a_software_extended_gamepad() {
        super::autoreleasepool(|_| {
            // SAFETY: documented GameController factory and getters; the
            // controller lives until the pool drains.
            unsafe {
                let ctrl: *mut AnyObject =
                    msg_send![class!(GCController), controllerWithExtendedGamepad];
                assert!(!ctrl.is_null());
                let gp: *mut AnyObject = msg_send![ctrl, extendedGamepad];
                assert!(!gp.is_null());
                assert!(!has_touchpad(gp));

                let idle = snapshot_of(gp);
                assert_eq!(idle, Default::default());

                press(msg_send![gp, buttonA], 1.0);
                press(msg_send![gp, buttonY], 1.0);
                press(msg_send![gp, rightShoulder], 1.0);
                press(msg_send![gp, buttonMenu], 1.0);
                press(msg_send![gp, leftTrigger], 0.5);
                push(msg_send![gp, leftThumbstick], -0.25, 0.75);
                push(msg_send![gp, rightThumbstick], 1.0, -1.0);
                push(msg_send![gp, dpad], 0.0, 1.0); // up

                let s = snapshot_of(gp);
                assert!(s.a && s.y && s.right_shoulder && s.menu && s.up);
                assert!(!s.b && !s.x && !s.left_shoulder && !s.down && !s.left && !s.right);
                assert!(!s.view && !s.touchpad);
                assert_eq!(s.left_trigger, 0.5);
                assert_eq!(s.right_trigger, 0.0);
                assert_eq!((s.left_x, s.left_y), (-0.25, 0.75));
                assert_eq!((s.right_x, s.right_y), (1.0, -1.0));
            }
        });
    }

    /// Smoke-tests linking `GameController.framework` and querying
    /// `[GCController controllers]` via `poll_controller` without panicking.
    #[test]
    fn poll_controller_queries_game_controller_framework_cleanly() {
        super::autoreleasepool(|_| {
            // SAFETY: plain GameController query.
            unsafe { super::poll_controller() };
            victory_hle::modules::gameinput::release_live_pad();
        });
    }
}
