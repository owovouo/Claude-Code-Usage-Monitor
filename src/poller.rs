use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::ffi::c_void;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use std::os::windows::process::CommandExt;

use crate::diagnose;
use crate::localization::Strings;
use crate::models::{AppUsageData, UsageData, UsageSection};

/// Codex's `/wham/usage` endpoint returns a sliding `reset_at` until the user
/// makes an actual API call. We run `codex exec .` once to lock the active
/// limit window and remember the previously observed reset across polls.
static CODEX_PREVIOUS_RESET_AT: Mutex<Option<SystemTime>> = Mutex::new(None);
static CODEX_LAST_TRIGGER_AT: Mutex<Option<Instant>> = Mutex::new(None);

/// When armed, the next `poll_codex` fires the Codex lock subprocess
/// unconditionally — bypassing the user toggle, Idle Hours, the
/// empty/sliding-window check, and the cooldown. Consumed on first poll so
/// subsequent polls behave normally.
///
/// Armed in two situations:
///   1. `--force-codex-trigger` CLI flag (debug / manual testing)
///   2. Exiting Idle Hours (so the lock runs immediately at the boundary,
///      without relying on the sliding-detect path which can miss-fire if
///      the boundary timer fires slightly before the wall-clock minute)
///
/// NOT armed on app startup, because the lock subprocess blocks the poll
/// thread for 30-120s and would leave the UI blank during that window. The
/// natural sliding-detect path fires the lock on the 2nd poll instead.
///
/// The caller in (2) must check `lock_codex_window` itself before arming —
/// this flag bypasses the user toggle.
static FORCE_CODEX_TRIGGER: AtomicBool = AtomicBool::new(false);

/// Arm a one-shot forced Codex trigger on the next poll.
pub fn arm_force_codex_trigger() {
    FORCE_CODEX_TRIGGER.store(true, Ordering::SeqCst);
}

/// Treat `reset_at` advancing by more than this between polls as evidence the
/// window is sliding (or has just rolled over). Both cases warrant a re-lock.
/// Kept small so 1-minute polls still detect sliding (where consecutive
/// reset_at values drift by ~60s with `now`). A locked window has drift = 0,
/// so 10s comfortably distinguishes the two while tolerating clock jitter.
const CODEX_SLIDING_TOLERANCE: Duration = Duration::from_secs(10);

/// Minimum time between two Codex triggers. Slightly less than the 5h window
/// length so a fresh trigger can fire shortly after each natural rollover.
const CODEX_TRIGGER_COOLDOWN: Duration = Duration::from_secs(4 * 3600);

/// A Codex 5h window counts as "anchored" once its `reset_at` is at most this
/// far in the future. A sliding (un-anchored) or just-rolled window always
/// reports `reset_at` ≈ now + 5h, so anything below the full window length is
/// a fixed, anchored reset that needs no (re-)lock. 5h − 60s grace = 17940s.
const CODEX_ANCHORED_MAX_REMAINING: Duration = Duration::from_secs(5 * 3600 - 60);
const CODEX_WEEKLY_ANCHORED_MAX_REMAINING: Duration =
    Duration::from_secs(7 * 24 * 3600 - 60);

/// `reset_at` of the most recently confirmed anchored window. The Codex
/// `/wham/usage` endpoint intermittently reports a sliding `reset_at` (≈ now+5h)
/// on scattered polls even while the window is actually anchored and idle, so a
/// single sliding reading cannot be trusted. We remember the last anchored
/// reading and treat the window as anchored until that time passes, ignoring
/// transient sliding flip-flops in between.
static CODEX_ANCHORED_UNTIL: Mutex<Option<SystemTime>> = Mutex::new(None);
static CODEX_WEEKLY_ANCHORED_UNTIL: Mutex<Option<SystemTime>> = Mutex::new(None);

static CLAUDE_CODE_LAST_TRIGGER_AT: Mutex<Option<Instant>> = Mutex::new(None);
const CLAUDE_CODE_TRIGGER_COOLDOWN: Duration = Duration::from_secs(4 * 3600);

const USAGE_URL: &str = "https://api.anthropic.com/api/oauth/usage";
const MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";
const CODEX_USAGE_URL: &str = "https://chatgpt.com/backend-api/wham/usage";
const ANTIGRAVITY_CREDENTIAL_TARGET: &str = "gemini:antigravity";
const ANTIGRAVITY_ENDPOINTS: &[&str] = &[
    "https://daily-cloudcode-pa.googleapis.com",
    "https://daily-cloudcode-pa.sandbox.googleapis.com",
    "https://cloudcode-pa.googleapis.com",
];
const CREATE_NO_WINDOW: u32 = 0x08000000;

const MODEL_FALLBACK_CHAIN: &[&str] = &["claude-3-haiku-20240307", "claude-haiku-4-5-20251001"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PollError {
    AuthRequired,
    NoCredentials,
    TokenExpired,
    RequestFailed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CredentialWatchMode {
    ActiveSource,
    AllSources,
    Antigravity,
}

pub type CredentialWatchSnapshot = Vec<String>;

#[derive(Deserialize)]
struct UsageResponse {
    five_hour: Option<UsageBucket>,
    seven_day: Option<UsageBucket>,
}

#[derive(Deserialize)]
struct UsageBucket {
    utilization: f64,
    resets_at: Option<String>,
}

#[derive(Deserialize)]
struct CodexAuthFile {
    tokens: Option<CodexTokenData>,
}

#[derive(Clone, Deserialize)]
struct CodexTokenData {
    access_token: String,
    account_id: Option<String>,
}

#[derive(Deserialize)]
struct CodexUsageResponse {
    rate_limit: Option<Option<Box<CodexRateLimitDetails>>>,
}

#[derive(Deserialize)]
struct CodexRateLimitDetails {
    primary_window: Option<Option<Box<CodexRateLimitWindow>>>,
    secondary_window: Option<Option<Box<CodexRateLimitWindow>>>,
}

#[derive(Deserialize)]
struct CodexRateLimitWindow {
    used_percent: f64,
    limit_window_seconds: Option<i64>,
    reset_at: i64,
}

#[derive(Deserialize)]
struct AntigravityAuthFile {
    token: AntigravityTokenData,
}

#[derive(Deserialize)]
struct AntigravityTokenData {
    access_token: String,
}

#[derive(Deserialize)]
struct AntigravityLoadResponse {
    #[serde(rename = "cloudaicompanionProject")]
    project: Option<String>,
}

#[derive(Deserialize)]
struct AntigravityModelsResponse {
    models: HashMap<String, AntigravityModelInfo>,
}

#[derive(Deserialize)]
struct AntigravityModelInfo {
    #[serde(rename = "quotaInfo")]
    quota_info: Option<AntigravityQuotaInfo>,
}

#[derive(Deserialize)]
struct AntigravityQuotaInfo {
    #[serde(rename = "remainingFraction")]
    remaining_fraction: Option<f64>,
    #[serde(rename = "resetTime")]
    reset_time: Option<String>,
}

#[derive(Deserialize)]
struct AntigravityQuotaSummaryResponse {
    groups: Option<Vec<AntigravityQuotaSummaryGroup>>,
}

#[derive(Deserialize)]
struct AntigravityQuotaSummaryGroup {
    #[serde(rename = "displayName")]
    display_name: Option<String>,
    description: Option<String>,
    buckets: Option<Vec<AntigravityQuotaSummaryBucket>>,
}

#[derive(Clone, Deserialize)]
struct AntigravityQuotaSummaryBucket {
    #[serde(rename = "bucketId")]
    bucket_id: Option<String>,
    #[serde(rename = "displayName")]
    display_name: Option<String>,
    window: Option<String>,
    #[serde(rename = "remainingFraction")]
    remaining_fraction: Option<f64>,
    #[serde(rename = "resetTime")]
    reset_time: Option<String>,
}

#[repr(C)]
struct CredentialW {
    flags: u32,
    type_: u32,
    target_name: *mut u16,
    comment: *mut u16,
    last_written: u64,
    credential_blob_size: u32,
    credential_blob: *mut u8,
    persist: u32,
    attribute_count: u32,
    attributes: *mut c_void,
    target_alias: *mut u16,
    user_name: *mut u16,
}

#[link(name = "Advapi32")]
extern "system" {
    fn CredReadW(
        target_name: *const u16,
        type_: u32,
        reserved_flags: u32,
        credential: *mut *mut CredentialW,
    ) -> i32;
    fn CredFree(buffer: *mut c_void);
}

pub fn poll(
    show_claude_code: bool,
    show_codex: bool,
    show_antigravity: bool,
    allow_codex_trigger: bool,
    allow_claude_trigger: bool,
) -> Result<AppUsageData, PollError> {
    poll_with(
        show_claude_code,
        show_codex,
        show_antigravity,
        || poll_claude_code(allow_claude_trigger),
        || poll_codex(allow_codex_trigger),
        poll_antigravity,
    )
}

fn poll_with(
    show_claude_code: bool,
    show_codex: bool,
    show_antigravity: bool,
    mut poll_claude_code: impl FnMut() -> Result<UsageData, PollError>,
    mut poll_codex: impl FnMut() -> Result<UsageData, PollError>,
    mut poll_antigravity: impl FnMut() -> Result<UsageData, PollError>,
) -> Result<AppUsageData, PollError> {
    let mut data = AppUsageData::default();
    let mut first_error = None;
    let active_provider_count = show_claude_code as u8 + show_codex as u8 + show_antigravity as u8;

    if show_claude_code {
        match poll_claude_code() {
            Ok(claude_code) => data.claude_code = Some(claude_code),
            Err(error) => {
                if active_provider_count > 1 {
                    diagnose::log(format!("Claude Code usage poll failed: {error:?}"));
                }
                first_error.get_or_insert(error);
            }
        }
    }

    if show_codex {
        match poll_codex() {
            Ok(codex) => data.codex = Some(codex),
            Err(error) => {
                if active_provider_count > 1 {
                    diagnose::log(format!("Codex usage poll failed: {error:?}"));
                }
                first_error.get_or_insert(error);
            }
        }
    }

    if show_antigravity {
        match poll_antigravity() {
            Ok(antigravity) => data.antigravity = Some(antigravity),
            Err(error) => {
                if active_provider_count > 1 {
                    diagnose::log(format!("Antigravity usage poll failed: {error:?}"));
                }
                first_error.get_or_insert(error);
            }
        }
    }

    if data.claude_code.is_none() && data.codex.is_none() && data.antigravity.is_none() {
        Err(first_error.unwrap_or(PollError::RequestFailed))
    } else {
        Ok(data)
    }
}

fn poll_claude_code(allow_trigger: bool) -> Result<UsageData, PollError> {
    let creds = match read_first_credentials() {
        Some(c) => c,
        None => {
            diagnose::log("poll failed: no Claude credentials found");
            return Err(PollError::NoCredentials);
        }
    };

    let creds = refresh_or_fallback(creds)?;
    let data = fetch_usage_with_fallback(&creds.access_token)?;

    if allow_trigger {
        let no_real_window = data.session.resets_at.is_none() || data.session.percentage <= 0.0;
        let cooldown_passed = CLAUDE_CODE_LAST_TRIGGER_AT
            .lock()
            .unwrap()
            .map(|t| t.elapsed() > CLAUDE_CODE_TRIGGER_COOLDOWN)
            .unwrap_or(true);

        diagnose::log(format!(
            "Claude Code poll decision: should_lock={} (no_resets_at={} pct={:.2} cooldown_passed={cooldown_passed} resets_at={:?})",
            no_real_window && cooldown_passed,
            data.session.resets_at.is_none(),
            data.session.percentage,
            data.session.resets_at
        ));

        if no_real_window && cooldown_passed {
            diagnose::log(format!(
                "Claude Code window needs lock (no_resets_at={} pct={:.2}); locking via API call",
                data.session.resets_at.is_none(),
                data.session.percentage
            ));
            *CLAUDE_CODE_LAST_TRIGGER_AT.lock().unwrap() = Some(Instant::now());
            if let Ok(locked) = fetch_usage_via_messages(&creds.access_token) {
                return Ok(locked);
            }
        }
    }

    Ok(data)
}

fn codex_active_window(
    data: &UsageData,
    five_hour_limit_unavailable: bool,
) -> (&UsageSection, Duration) {
    if five_hour_limit_unavailable {
        (&data.weekly, CODEX_WEEKLY_ANCHORED_MAX_REMAINING)
    } else {
        (&data.session, CODEX_ANCHORED_MAX_REMAINING)
    }
}

fn codex_reset_is_sliding(reset_at: Option<SystemTime>, max_remaining: Duration) -> bool {
    reset_at
        .and_then(|t| t.duration_since(SystemTime::now()).ok())
        .map(|remaining| remaining > max_remaining)
        .unwrap_or(true)
}

/// Records a fixed reset while ignoring sliding readings near the full window.
fn note_codex_anchor(
    reset_at: Option<SystemTime>,
    max_remaining: Duration,
    anchored_until: &Mutex<Option<SystemTime>>,
) {
    if let Some(remaining) = reset_at.and_then(|t| t.duration_since(SystemTime::now()).ok()) {
        if remaining <= max_remaining {
            *anchored_until.lock().unwrap() = reset_at;
        }
    }
}

fn should_trigger_codex(
    force: bool,
    window_still_anchored: bool,
    no_real_window: bool,
    is_sliding: bool,
    cooldown_passed: bool,
) -> bool {
    (no_real_window || !window_still_anchored)
        && (force || ((no_real_window || is_sliding) && cooldown_passed))
}

fn poll_codex(allow_trigger: bool) -> Result<UsageData, PollError> {
    let creds = match read_codex_credentials() {
        Some(creds) => creds,
        None => {
            diagnose::log("Codex usage poll failed: no Codex credentials found");
            return Err(PollError::NoCredentials);
        }
    };

    let (data, five_hour_limit_unavailable) =
        match fetch_codex_usage(&creds.access_token, creds.account_id.as_deref()) {
            Ok(result) => result,
            Err(PollError::AuthRequired) => {
                // Token expired — refresh it. The subprocess may incidentally
                // lock the 5h window as a side effect, but we do NOT set the
                // cooldown here: whether or not it anchored will be detected by
                // the normal sliding-check on subsequent polls, which will fire
                // the trigger path (with 3-minute anchor verification) if needed.
                cli_refresh_codex_token();
                let refreshed = read_codex_credentials().ok_or(PollError::TokenExpired)?;
                fetch_codex_usage(&refreshed.access_token, refreshed.account_id.as_deref())?
            }
            Err(error) => return Err(error),
        };

    let (window, max_remaining) =
        codex_active_window(&data, five_hour_limit_unavailable);
    let anchored_until = if five_hour_limit_unavailable {
        &CODEX_WEEKLY_ANCHORED_UNTIL
    } else {
        &CODEX_ANCHORED_UNTIL
    };
    let current = window.resets_at;

    // Record an anchored reading (if this poll shows one) before deciding
    // whether to lock, so the trigger logic can ignore later sliding flip-flops.
    note_codex_anchor(current, max_remaining, anchored_until);

    // Decide whether to lock a window. Trigger conditions (all gated by the
    // cooldown so a stuck/failed trigger can't loop, and by the caller's
    // `allow_trigger`, which is false during Idle Hours):
    //
    //   1. `resets_at` is None               → API reports no window at all
    //   2. 5h `percentage <= 1.0`            → session window is empty
    //   3. `reset_at` advanced > tolerance   → cross-poll evidence of sliding
    //   4. weekly reset is still nearly +7d  → weekly window is not anchored
    //
    // All of these are suppressed while the last confirmed anchored window is
    // still in the future (`window_still_anchored`). The Codex endpoint
    // intermittently reports a sliding `reset_at` (≈ now+5h) on scattered polls
    // even on an anchored, idle window, so reacting to a single sliding reading
    // would fire a useless lock — and that lock's cooldown would then block the
    // real re-lock when the window genuinely expires.
    //
    // `--force-codex-trigger` (debug) and the idle-exit force still respect the
    // anchored guard — there is no point re-locking an already-anchored window.
    let force = FORCE_CODEX_TRIGGER.swap(false, Ordering::SeqCst);
    if allow_trigger || force {
        let previous = *CODEX_PREVIOUS_RESET_AT.lock().unwrap();

        // The 5h endpoint reports an unused window as 1%. Weekly usage can
        // remain at 0–1% after a successful lock, so use its moving reset time.
        let no_real_window = current.is_none()
            || (!five_hour_limit_unavailable && window.percentage <= 1.0);
        let near_full_weekly_window = five_hour_limit_unavailable
            && codex_reset_is_sliding(current, max_remaining);
        let is_sliding = near_full_weekly_window
            || match (previous, current) {
                (Some(prev), Some(curr)) => curr
                    .duration_since(prev)
                    .map(|d| d > CODEX_SLIDING_TOLERANCE)
                    .unwrap_or(false),
                _ => false,
            };

        // Trust the remembered anchor over the current single reading: while the
        // last anchored reset is still in the future the window is anchored and
        // needs no lock, regardless of transient sliding flip-flops. Once it
        // passes (or was never set), triggering is allowed again.
        let window_still_anchored = anchored_until
            .lock()
            .unwrap()
            .map(|t| t > SystemTime::now())
            .unwrap_or(false);

        let cooldown_passed = CODEX_LAST_TRIGGER_AT
            .lock()
            .unwrap()
            .map(|t| t.elapsed() > CODEX_TRIGGER_COOLDOWN)
            .unwrap_or(true);

        let should_trigger = should_trigger_codex(
            force,
            window_still_anchored,
            no_real_window,
            is_sliding,
            cooldown_passed,
        );

        diagnose::log(format!(
            "Codex poll decision: should_trigger={should_trigger} (force={force} five_hour_unavailable={five_hour_limit_unavailable} anchored={window_still_anchored} no_window={no_real_window} sliding={is_sliding} cooldown_passed={cooldown_passed} pct={:.2} prev={previous:?} curr={current:?})",
            window.percentage
        ));

        if should_trigger {
            diagnose::log(format!(
                "Codex trigger (force={force} no_window={no_real_window} sliding={is_sliding} cooldown_passed={cooldown_passed} pct={:.2} prev={previous:?} curr={current:?}); running Codex lock subprocess",
                window.percentage
            ));
            *CODEX_LAST_TRIGGER_AT.lock().unwrap() = Some(Instant::now());
            // Run the lock subprocess in a background thread so the poll
            // returns immediately and the UI stays responsive. The cooldown
            // is already set above, so subsequent polls won't re-trigger.
            // The next regular poll will observe the anchored reset_at.
            std::thread::spawn(|| {
                cli_refresh_codex_token();

                // Wait 3 minutes, then verify that the selected 5h or 7d reset
                // is now fixed rather than still a full window from now.
                //
                // If still sliding: reset the cooldown so the next regular
                // poll can retry immediately, rather than waiting 4 hours.
                std::thread::sleep(Duration::from_secs(180));

                match read_codex_credentials()
                    .ok_or(PollError::NoCredentials)
                    .and_then(|c| {
                        fetch_codex_usage(&c.access_token, c.account_id.as_deref())
                    })
                {
                    Ok((data, five_hour_limit_unavailable)) => {
                        let (window, max_remaining) =
                            codex_active_window(&data, five_hour_limit_unavailable);
                        let anchored_until = if five_hour_limit_unavailable {
                            &CODEX_WEEKLY_ANCHORED_UNTIL
                        } else {
                            &CODEX_ANCHORED_UNTIL
                        };
                        let still_sliding =
                            codex_reset_is_sliding(window.resets_at, max_remaining);

                        diagnose::log(format!(
                            "Codex anchor verify (3m): five_hour_unavailable={five_hour_limit_unavailable} pct={:.2} resets_at={:?} still_sliding={still_sliding}",
                            window.percentage, window.resets_at
                        ));

                        if still_sliding {
                            // Lock subprocess didn't anchor the window.
                            // Reset the cooldown so the next poll retries.
                            *CODEX_LAST_TRIGGER_AT.lock().unwrap() = None;
                            diagnose::log(
                                "Codex lock didn't anchor; cooldown reset, will retry on next poll",
                            );
                        } else {
                            // Anchor confirmed — remember it so later sliding
                            // flip-flop readings don't re-trigger a lock.
                            note_codex_anchor(
                                window.resets_at,
                                max_remaining,
                                anchored_until,
                            );
                        }
                    }
                    Err(e) => {
                        diagnose::log(format!(
                            "Codex anchor verify failed (3m): {e:?}; resetting cooldown for retry"
                        ));
                        *CODEX_LAST_TRIGGER_AT.lock().unwrap() = None;
                    }
                }
            });
        }
    }

    // Always record the reset_at we just observed so the next poll can detect
    // sliding behavior.
    *CODEX_PREVIOUS_RESET_AT.lock().unwrap() = current;

    Ok(data)
}

fn poll_antigravity() -> Result<UsageData, PollError> {
    let creds = match read_antigravity_credentials() {
        Some(creds) => creds,
        None => {
            diagnose::log("Antigravity usage poll failed: no Antigravity credentials found");
            return Err(PollError::NoCredentials);
        }
    };

    fetch_antigravity_usage(&creds.access_token)
}

fn refresh_or_fallback(mut creds: Credentials) -> Result<Credentials, PollError> {
    loop {
        if !is_token_expired(creds.expires_at) {
            return Ok(creds);
        }

        let source = creds.source.clone();
        cli_refresh_token(&source);

        match read_credentials_from_source(&source) {
            Some(refreshed) if !is_token_expired(refreshed.expires_at) => return Ok(refreshed),
            Some(_) => diagnose::log(format!(
                "credentials from {source:?} still expired after refresh attempt"
            )),
            None => diagnose::log(format!(
                "credentials from {source:?} unavailable after refresh attempt"
            )),
        }

        match read_next_credentials_after(&source) {
            Some(next) => creds = next,
            None => return Err(PollError::TokenExpired),
        }
    }
}

/// Invoke the Claude CLI with a minimal prompt to force its internal
/// OAuth token refresh.
fn cli_refresh_token(source: &CredentialSource) {
    match source {
        CredentialSource::Windows(_) => cli_refresh_windows_token(),
        CredentialSource::Wsl { distro } => cli_refresh_wsl_token(distro),
    }
}

fn cli_refresh_windows_token() {
    let claude_path = resolve_windows_claude_path();
    let is_cmd = claude_path.to_lowercase().ends_with(".cmd");
    diagnose::log(format!(
        "attempting Windows Claude token refresh via {claude_path}"
    ));

    let args: &[&str] = &["-p", "."];

    let mut cmd = if is_cmd {
        let mut c = Command::new("cmd.exe");
        c.arg("/c").arg(&claude_path).args(args);
        c
    } else {
        let mut c = Command::new(&claude_path);
        c.args(args);
        c
    };
    cmd.env_remove("CLAUDECODE")
        .env_remove("CLAUDE_CODE_ENTRYPOINT")
        .creation_flags(CREATE_NO_WINDOW)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(error) => {
            diagnose::log_error("unable to spawn Windows Claude token refresh", error);
            return;
        }
    };

    // Wait up to 30 seconds — don't block the poll thread forever
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if start.elapsed() > Duration::from_secs(30) {
                    let _ = child.kill();
                    break;
                }
                std::thread::sleep(Duration::from_millis(500));
            }
            Err(_) => break,
        }
    }
}

fn cli_refresh_wsl_token(distro: &str) {
    diagnose::log(format!(
        "attempting WSL Claude token refresh in distro {distro}"
    ));
    let mut cmd = Command::new("wsl.exe");
    cmd.arg("-d")
        .arg(distro)
        .arg("--")
        .arg("bash")
        .arg("-lic")
        .arg("if command -v claude >/dev/null 2>&1; then claude -p .; elif [ -x \"$HOME/.local/bin/claude\" ]; then \"$HOME/.local/bin/claude\" -p .; else exit 127; fi")
        .env_remove("CLAUDECODE")
        .env_remove("CLAUDE_CODE_ENTRYPOINT")
        .creation_flags(CREATE_NO_WINDOW)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(error) => {
            diagnose::log_error("unable to spawn WSL Claude token refresh", error);
            return;
        }
    };

    wait_for_refresh(&mut child);
}

fn cli_refresh_codex_token() {
    let codex_path = resolve_windows_codex_path();
    let is_cmd = codex_path.to_lowercase().ends_with(".cmd");
    let is_ps1 = codex_path.to_lowercase().ends_with(".ps1");
    diagnose::log(format!(
        "attempting Windows Codex token refresh via {codex_path}"
    ));

    // Locking the Codex 5h window requires the API to attribute the call to
    // a real session. Previous attempts with --ephemeral / --ignore-* flags
    // burned only ~2.5k tokens and never anchored anything — likely below
    // whatever noise threshold the API uses. GitHub openai/codex#19996
    // reports normal CLI startup alone consumes 21-43k tokens (loading user
    // config, AGENTS.md, rules, etc.). We now let all of that load so the
    // call registers as a real Codex session.
    //
    // Cost: ~25-50k tokens per lock. Runtime: 30-60s.
    //
    // Flags:
    //   --skip-git-repo-check : bypass git-trust requirement
    //   -s read-only          : sandbox, no shell execution (still mandatory)
    let args: &[&str] = &[
        "exec",
        "--skip-git-repo-check",
        "-s",
        "read-only",
        "ok",
    ];

    let mut cmd = if is_cmd {
        let mut c = Command::new("cmd.exe");
        c.arg("/c").arg(&codex_path).args(args);
        c
    } else if is_ps1 {
        let mut c = Command::new("powershell.exe");
        c.arg("-NoProfile")
            .arg("-ExecutionPolicy")
            .arg("Bypass")
            .arg("-File")
            .arg(&codex_path)
            .args(args);
        c
    } else {
        let mut c = Command::new(&codex_path);
        c.args(args);
        c
    };
    cmd.creation_flags(CREATE_NO_WINDOW)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    // 180s gives the subprocess room to fully complete (session init + "ok"
    // response is typically 30-90s without --ephemeral). Hitting the timeout
    // is not a failure per se: empirically the API call that anchors the 5h
    // window registers within the first few seconds, so the lock can still
    // succeed even if we kill the process mid-stream.
    let started = Instant::now();
    let output = match run_with_timeout(&mut cmd, Duration::from_secs(180)) {
        Some(output) => output,
        None => {
            diagnose::log(format!(
                "codex lock subprocess exceeded 180s and was killed after {:?} \
                 (window may still have anchored — check post-trigger reset_at)",
                started.elapsed()
            ));
            return;
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    diagnose::log(format!(
        "codex lock subprocess finished: status={:?} after {:?}\n--- stdout ({} bytes) ---\n{}\n--- stderr ({} bytes) ---\n{}\n--- end ---",
        output.status.code(),
        started.elapsed(),
        output.stdout.len(),
        stdout.trim_end(),
        output.stderr.len(),
        stderr.trim_end()
    ));
}

/// Spawn a command and wait up to `timeout` for it to finish.
/// Returns None if the process fails to start or exceeds the deadline.
fn run_with_timeout(cmd: &mut Command, timeout: Duration) -> Option<std::process::Output> {
    let mut child = cmd.spawn().ok()?;
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => return child.wait_with_output().ok(),
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(_) => return None,
        }
    }
}

#[derive(Debug)]
#[allow(dead_code)]
enum RefreshOutcome {
    Exited(Option<i32>),
    KilledByTimeout,
    WaitFailed,
}

fn wait_for_refresh(child: &mut std::process::Child) -> RefreshOutcome {
    // Wait up to 30 seconds; don't block the poll thread forever.
    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return RefreshOutcome::Exited(status.code()),
            Ok(None) => {
                if start.elapsed() > Duration::from_secs(30) {
                    let _ = child.kill();
                    return RefreshOutcome::KilledByTimeout;
                }
                std::thread::sleep(Duration::from_millis(500));
            }
            Err(_) => return RefreshOutcome::WaitFailed,
        }
    }
}

/// Resolve the full path to the `claude` CLI executable.
fn resolve_windows_claude_path() -> String {
    for name in &["claude.cmd", "claude"] {
        if Command::new(name)
            .arg("--version")
            .creation_flags(CREATE_NO_WINDOW)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok()
        {
            return name.to_string();
        }
    }

    for name in &["claude.cmd", "claude"] {
        if let Ok(output) = Command::new("where.exe")
            .arg(name)
            .creation_flags(CREATE_NO_WINDOW)
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(first_line) = stdout.lines().next() {
                    let path = first_line.trim().to_string();
                    if !path.is_empty() {
                        return path;
                    }
                }
            }
        }
    }

    "claude.cmd".to_string()
}

fn resolve_windows_codex_path() -> String {
    for name in &["codex.cmd", "codex.ps1", "codex.exe", "codex"] {
        if Command::new(name)
            .arg("--version")
            .creation_flags(CREATE_NO_WINDOW)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok()
        {
            return name.to_string();
        }
    }

    for name in &["codex.cmd", "codex.ps1", "codex.exe", "codex"] {
        if let Ok(output) = Command::new("where.exe")
            .arg(name)
            .creation_flags(CREATE_NO_WINDOW)
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(first_line) = stdout.lines().next() {
                    let path = first_line.trim().to_string();
                    if !path.is_empty() {
                        return path;
                    }
                }
            }
        }
    }

    "codex.cmd".to_string()
}

fn build_agent() -> Result<ureq::Agent, PollError> {
    let tls = native_tls::TlsConnector::new().map_err(|_| PollError::RequestFailed)?;
    Ok(ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(30))
        .tls_connector(std::sync::Arc::new(tls))
        .build())
}

pub fn credential_watch_snapshot(mode: CredentialWatchMode) -> CredentialWatchSnapshot {
    if mode == CredentialWatchMode::Antigravity {
        return vec![antigravity_credential_watch_signature()];
    }

    let sources = match mode {
        CredentialWatchMode::ActiveSource => read_first_credentials()
            .map(|creds| vec![creds.source])
            .unwrap_or_else(all_known_credential_sources),
        CredentialWatchMode::AllSources => all_known_credential_sources(),
        CredentialWatchMode::Antigravity => unreachable!(),
    };

    let mut snapshot: CredentialWatchSnapshot = sources
        .into_iter()
        .filter_map(|source| credential_watch_signature(&source))
        .collect();
    snapshot.sort();
    snapshot.dedup();
    snapshot
}

fn all_known_credential_sources() -> Vec<CredentialSource> {
    let mut sources = Vec::new();
    if let Some(source) = windows_credential_source() {
        sources.push(source);
    }
    for distro in list_wsl_distros() {
        sources.push(CredentialSource::Wsl { distro });
    }
    sources
}

fn windows_credential_source() -> Option<CredentialSource> {
    let home = dirs::home_dir()?;
    Some(CredentialSource::Windows(
        home.join(".claude").join(".credentials.json"),
    ))
}

fn credential_watch_signature(source: &CredentialSource) -> Option<String> {
    match source {
        CredentialSource::Windows(path) => Some(windows_credential_watch_signature(path)),
        CredentialSource::Wsl { distro } => wsl_credential_watch_signature(distro),
    }
}

fn windows_credential_watch_signature(path: &PathBuf) -> String {
    let key = format!("win:{}", path.display());
    match std::fs::metadata(path) {
        Ok(metadata) => {
            let modified = metadata
                .modified()
                .ok()
                .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
                .map(|value| value.as_secs())
                .unwrap_or(0);
            format!("{key}|present|{}|{modified}", metadata.len())
        }
        Err(_) => format!("{key}|missing"),
    }
}

fn wsl_credential_watch_signature(distro: &str) -> Option<String> {
    let output = run_with_timeout(
        Command::new("wsl.exe")
            .arg("-d")
            .arg(distro)
            .arg("--")
            .arg("sh")
            .arg("-lc")
            .arg(
                "if [ -f ~/.claude/.credentials.json ]; then \
                 stat -c 'present|%s|%Y' ~/.claude/.credentials.json; \
                 else echo missing; fi",
            )
            .creation_flags(CREATE_NO_WINDOW)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null()),
        Duration::from_secs(5),
    )?;

    let state = if output.status.success() {
        decode_wsl_text(&output.stdout).trim().to_string()
    } else {
        format!("status-{}", output.status)
    };

    Some(format!("wsl:{distro}|{state}"))
}

fn fetch_usage_with_fallback(token: &str) -> Result<UsageData, PollError> {
    // Try the dedicated usage endpoint first
    match try_usage_endpoint(token)? {
        Some(data) => {
            // If reset timers are missing, fill them in from the Messages API
            if data.session.resets_at.is_none() || data.weekly.resets_at.is_none() {
                if let Ok(fallback) = fetch_usage_via_messages(token) {
                    let mut merged = data;
                    if merged.session.resets_at.is_none() {
                        merged.session.resets_at = fallback.session.resets_at;
                    }
                    if merged.weekly.resets_at.is_none() {
                        merged.weekly.resets_at = fallback.weekly.resets_at;
                    }
                    return Ok(merged);
                }
            }
            return Ok(data);
        }
        None => {}
    }

    // Fall back to Messages API with rate limit headers
    let result = fetch_usage_via_messages(token);
    if result.is_err() {
        diagnose::log("usage endpoint and Messages API fallback both failed");
    }
    result
}

fn try_usage_endpoint(token: &str) -> Result<Option<UsageData>, PollError> {
    let agent = build_agent()?;

    let resp = match agent
        .get(USAGE_URL)
        .set("Authorization", &format!("Bearer {token}"))
        .set("anthropic-beta", "oauth-2025-04-20")
        .call()
    {
        Ok(resp) => resp,
        Err(ureq::Error::Status(code, _)) if code == 401 || code == 403 => {
            diagnose::log(format!(
                "usage endpoint returned auth error status {code}; re-login required"
            ));
            return Err(PollError::AuthRequired);
        }
        Err(_) => return Ok(None),
    };

    let response: UsageResponse = match resp.into_json() {
        Ok(response) => response,
        Err(_) => return Ok(None),
    };
    let mut data = UsageData::default();

    if let Some(bucket) = &response.five_hour {
        data.session.percentage = bucket.utilization;
        data.session.resets_at = parse_iso8601(bucket.resets_at.as_deref());
    }

    if let Some(bucket) = &response.seven_day {
        data.weekly.percentage = bucket.utilization;
        data.weekly.resets_at = parse_iso8601(bucket.resets_at.as_deref());
    }

    Ok(Some(data))
}

fn fetch_usage_via_messages(token: &str) -> Result<UsageData, PollError> {
    let agent = build_agent()?;

    for model in MODEL_FALLBACK_CHAIN {
        let body = serde_json::json!({
            "model": model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "."}]
        });

        let response = match agent
            .post(MESSAGES_URL)
            .set("Authorization", &format!("Bearer {token}"))
            .set("anthropic-version", "2023-06-01")
            .set("anthropic-beta", "oauth-2025-04-20")
            .send_json(&body)
        {
            Ok(resp) => resp,
            Err(ureq::Error::Status(code, _)) if code == 401 || code == 403 => {
                diagnose::log(format!(
                    "messages endpoint returned auth error status {code}; re-login required"
                ));
                return Err(PollError::AuthRequired);
            }
            Err(ureq::Error::Status(_code, resp)) => resp,
            Err(_) => continue,
        };

        let h5 = response.header("anthropic-ratelimit-unified-5h-utilization");
        let h7 = response.header("anthropic-ratelimit-unified-7d-utilization");
        let hs = response.header("anthropic-ratelimit-unified-status");

        if h5.is_some() || h7.is_some() || hs.is_some() {
            return Ok(parse_rate_limit_headers(&response));
        }
    }

    Err(PollError::RequestFailed)
}

fn parse_rate_limit_headers(response: &ureq::Response) -> UsageData {
    let mut data = UsageData::default();

    data.session.percentage =
        get_header_f64(response, "anthropic-ratelimit-unified-5h-utilization") * 100.0;
    data.session.resets_at = unix_to_system_time(get_header_i64(
        response,
        "anthropic-ratelimit-unified-5h-reset",
    ));

    data.weekly.percentage =
        get_header_f64(response, "anthropic-ratelimit-unified-7d-utilization") * 100.0;
    data.weekly.resets_at = unix_to_system_time(get_header_i64(
        response,
        "anthropic-ratelimit-unified-7d-reset",
    ));

    let overall_reset = get_header_i64(response, "anthropic-ratelimit-unified-reset");

    if data.session.percentage == 0.0 && data.weekly.percentage == 0.0 {
        let status = response.header("anthropic-ratelimit-unified-status");
        if status == Some("rejected") {
            let claim = response.header("anthropic-ratelimit-unified-representative-claim");
            match claim {
                Some("five_hour") => data.session.percentage = 100.0,
                Some("seven_day") => data.weekly.percentage = 100.0,
                _ => {}
            }
        }

        if data.session.resets_at.is_none() && overall_reset.is_some() {
            data.session.resets_at = unix_to_system_time(overall_reset);
        }
    }

    data
}

fn fetch_codex_usage(
    token: &str,
    account_id: Option<&str>,
) -> Result<(UsageData, bool), PollError> {
    let agent = build_agent()?;
    let mut request = agent
        .get(CODEX_USAGE_URL)
        .set("Authorization", &format!("Bearer {token}"))
        .set("User-Agent", "codex-cli");

    if let Some(account_id) = account_id.filter(|value| !value.is_empty()) {
        request = request.set("ChatGPT-Account-Id", account_id);
    }

    let resp = match request.call() {
        Ok(resp) => resp,
        Err(ureq::Error::Status(code, _)) if code == 401 || code == 403 => {
            diagnose::log(format!(
                "Codex usage endpoint returned auth error status {code}; refresh required"
            ));
            return Err(PollError::AuthRequired);
        }
        Err(error) => {
            diagnose::log_error("Codex usage endpoint request failed", error);
            return Err(PollError::RequestFailed);
        }
    };

    let response: CodexUsageResponse = match resp.into_json() {
        Ok(response) => response,
        Err(error) => {
            diagnose::log_error("unable to parse Codex usage response", error);
            return Err(PollError::RequestFailed);
        }
    };

    Ok(codex_usage_from_response(response))
}

/// Parse a `/wham/usage` response into our internal `UsageData` shape.
///
/// The boolean is true when the API explicitly puts a 7d window in the primary
/// slot, so polling and lock verification target that window instead of 5h.
/// A missing primary window keeps the old lock behavior for inactive 5h windows.
fn codex_usage_from_response(response: CodexUsageResponse) -> (UsageData, bool) {
    let mut data = UsageData::default();
    let Some(details_box) = response.rate_limit.flatten() else {
        return (data, false);
    };
    let details = *details_box;
    let mut five_hour_limit_unavailable = false;

    if let Some(window) = details.primary_window.flatten() {
        if window.limit_window_seconds == Some(7 * 24 * 60 * 60) {
            five_hour_limit_unavailable = true;
            data.weekly = codex_section_from_window(&window);
        } else {
            data.session = codex_section_from_window(&window);
        }
    }

    if let Some(window) = details.secondary_window.flatten() {
        data.weekly = codex_section_from_window(&window);
    }

    (data, five_hour_limit_unavailable)
}

fn codex_section_from_window(window: &CodexRateLimitWindow) -> UsageSection {
    UsageSection {
        percentage: window.used_percent,
        resets_at: unix_to_system_time(Some(window.reset_at)),
    }
}

fn antigravity_credential_watch_signature() -> String {
    let Some(content) = read_windows_generic_credential(ANTIGRAVITY_CREDENTIAL_TARGET) else {
        return format!("{ANTIGRAVITY_CREDENTIAL_TARGET}|missing");
    };

    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    format!(
        "{ANTIGRAVITY_CREDENTIAL_TARGET}|present|{}|{}",
        content.len(),
        hasher.finish()
    )
}

fn fetch_antigravity_usage(token: &str) -> Result<UsageData, PollError> {
    let mut auth_error = false;
    let mut last_error = PollError::RequestFailed;

    for base_url in ANTIGRAVITY_ENDPOINTS {
        match fetch_antigravity_usage_from_endpoint(base_url, token) {
            Ok(data) => return Ok(data),
            Err(PollError::AuthRequired) => auth_error = true,
            Err(error) => last_error = error,
        }
    }

    if auth_error {
        Err(PollError::AuthRequired)
    } else {
        Err(last_error)
    }
}

fn fetch_antigravity_usage_from_endpoint(
    base_url: &str,
    token: &str,
) -> Result<UsageData, PollError> {
    let project = fetch_antigravity_project(base_url, token)?;
    if let Some(project) = project.as_deref() {
        match fetch_antigravity_quota_summary(base_url, token, project) {
            Ok(data) => return Ok(data),
            Err(PollError::AuthRequired) => return Err(PollError::AuthRequired),
            Err(error) => diagnose::log(format!(
                "Antigravity retrieveUserQuotaSummary failed, falling back to model quota: {error:?}"
            )),
        }
    }

    let session = fetch_antigravity_model_quota(base_url, token, project.as_deref())?;
    let weekly = UsageSection::default();

    Ok(UsageData { session, weekly })
}

fn fetch_antigravity_project(base_url: &str, token: &str) -> Result<Option<String>, PollError> {
    let agent = build_agent()?;
    let body = serde_json::json!({
        "metadata": {
            "ideType": "ANTIGRAVITY"
        }
    });

    let resp = match agent
        .post(&format!("{base_url}/v1internal:loadCodeAssist"))
        .set("Authorization", &format!("Bearer {token}"))
        .set("Content-Type", "application/json")
        .set("User-Agent", "antigravity")
        .send_json(&body)
    {
        Ok(resp) => resp,
        Err(ureq::Error::Status(code, _)) if code == 401 || code == 403 => {
            diagnose::log(format!(
                "Antigravity loadCodeAssist returned auth error status {code}"
            ));
            return Err(PollError::AuthRequired);
        }
        Err(error) => {
            diagnose::log_error("Antigravity loadCodeAssist request failed", error);
            return Err(PollError::RequestFailed);
        }
    };

    let response: AntigravityLoadResponse = match resp.into_json() {
        Ok(response) => response,
        Err(error) => {
            diagnose::log_error("unable to parse Antigravity loadCodeAssist response", error);
            return Err(PollError::RequestFailed);
        }
    };

    Ok(response.project.filter(|project| !project.is_empty()))
}

fn fetch_antigravity_model_quota(
    base_url: &str,
    token: &str,
    project: Option<&str>,
) -> Result<UsageSection, PollError> {
    let agent = build_agent()?;
    let body = match project {
        Some(project) => serde_json::json!({ "project": project }),
        None => serde_json::json!({}),
    };

    let resp = match agent
        .post(&format!("{base_url}/v1internal:fetchAvailableModels"))
        .set("Authorization", &format!("Bearer {token}"))
        .set("Content-Type", "application/json")
        .set("User-Agent", "antigravity")
        .send_json(&body)
    {
        Ok(resp) => resp,
        Err(ureq::Error::Status(code, _)) if code == 401 || code == 403 => {
            diagnose::log(format!(
                "Antigravity fetchAvailableModels returned auth error status {code}"
            ));
            return Err(PollError::AuthRequired);
        }
        Err(error) => {
            diagnose::log_error("Antigravity fetchAvailableModels request failed", error);
            return Err(PollError::RequestFailed);
        }
    };

    let response: AntigravityModelsResponse = match resp.into_json() {
        Ok(response) => response,
        Err(error) => {
            diagnose::log_error(
                "unable to parse Antigravity fetchAvailableModels response",
                error,
            );
            return Err(PollError::RequestFailed);
        }
    };

    best_antigravity_section(response.models.into_iter().filter_map(|(model, info)| {
        let quota = info.quota_info?;
        if !is_antigravity_display_model(&model) {
            return None;
        }
        antigravity_section_from_quota(quota)
    }))
    .ok_or(PollError::RequestFailed)
}

fn fetch_antigravity_quota_summary(
    base_url: &str,
    token: &str,
    project: &str,
) -> Result<UsageData, PollError> {
    let agent = build_agent()?;
    let body = serde_json::json!({ "project": project });

    let resp = match agent
        .post(&format!("{base_url}/v1internal:retrieveUserQuotaSummary"))
        .set("Authorization", &format!("Bearer {token}"))
        .set("Content-Type", "application/json")
        .set("User-Agent", "antigravity")
        .send_json(&body)
    {
        Ok(resp) => resp,
        Err(ureq::Error::Status(code, _)) if code == 401 || code == 403 => {
            return Err(PollError::AuthRequired);
        }
        Err(error) => {
            diagnose::log_error("Antigravity retrieveUserQuotaSummary request failed", error);
            return Err(PollError::RequestFailed);
        }
    };

    let response: AntigravityQuotaSummaryResponse = match resp.into_json() {
        Ok(response) => response,
        Err(error) => {
            diagnose::log_error(
                "unable to parse Antigravity retrieveUserQuotaSummary response",
                error,
            );
            return Err(PollError::RequestFailed);
        }
    };

    antigravity_usage_from_summary(response).ok_or(PollError::RequestFailed)
}

fn antigravity_section_from_quota(quota: AntigravityQuotaInfo) -> Option<UsageSection> {
    let remaining = quota.remaining_fraction?.clamp(0.0, 1.0);
    Some(UsageSection {
        percentage: (1.0 - remaining) * 100.0,
        resets_at: parse_iso8601(quota.reset_time.as_deref()),
    })
}

fn antigravity_section_from_summary_bucket(
    bucket: &AntigravityQuotaSummaryBucket,
) -> Option<UsageSection> {
    let remaining = bucket.remaining_fraction?.clamp(0.0, 1.0);
    Some(UsageSection {
        percentage: (1.0 - remaining) * 100.0,
        resets_at: parse_iso8601(bucket.reset_time.as_deref()),
    })
}

fn antigravity_usage_from_summary(response: AntigravityQuotaSummaryResponse) -> Option<UsageData> {
    let mut fallback = None;

    for group in response.groups.unwrap_or_default() {
        let is_gemini = is_antigravity_gemini_summary_group(&group);
        let usage = antigravity_usage_from_summary_group(group);

        if is_gemini && usage.is_some() {
            return usage;
        }

        if fallback.is_none() {
            fallback = usage;
        }
    }

    fallback
}

fn antigravity_usage_from_summary_group(group: AntigravityQuotaSummaryGroup) -> Option<UsageData> {
    let mut data = UsageData::default();
    let mut has_quota = false;

    for bucket in group.buckets.unwrap_or_default() {
        let Some(section) = antigravity_section_from_summary_bucket(&bucket) else {
            continue;
        };

        match bucket.window.as_deref() {
            Some(window) if window.eq_ignore_ascii_case("5h") => {
                data.session = section;
                has_quota = true;
            }
            Some(window) if window.eq_ignore_ascii_case("weekly") => {
                data.weekly = section;
                has_quota = true;
            }
            _ => {}
        }
    }

    has_quota.then_some(data)
}

fn is_antigravity_gemini_summary_group(group: &AntigravityQuotaSummaryGroup) -> bool {
    group
        .display_name
        .as_deref()
        .is_some_and(|name| name.to_ascii_lowercase().contains("gemini"))
        || group
            .description
            .as_deref()
            .is_some_and(|description| description.to_ascii_lowercase().contains("gemini"))
        || group.buckets.as_ref().is_some_and(|buckets| {
            buckets.iter().any(|bucket| {
                bucket
                    .bucket_id
                    .as_deref()
                    .is_some_and(|id| id.to_ascii_lowercase().starts_with("gemini-"))
                    || bucket
                        .display_name
                        .as_deref()
                        .is_some_and(|name| name.to_ascii_lowercase().contains("gemini"))
            })
        })
}

fn best_antigravity_section<I>(sections: I) -> Option<UsageSection>
where
    I: IntoIterator<Item = UsageSection>,
{
    sections.into_iter().max_by(|a, b| {
        a.percentage
            .partial_cmp(&b.percentage)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.resets_at.cmp(&b.resets_at))
    })
}

fn is_antigravity_display_model(model: &str) -> bool {
    model.starts_with("gemini")
        || model.starts_with("claude")
        || model.starts_with("gpt")
        || model.starts_with("image")
        || model.starts_with("imagen")
}

fn get_header_f64(response: &ureq::Response, name: &str) -> f64 {
    response
        .header(name)
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0)
}

fn get_header_i64(response: &ureq::Response, name: &str) -> Option<i64> {
    response.header(name).and_then(|s| s.parse::<i64>().ok())
}

fn unix_to_system_time(unix_secs: Option<i64>) -> Option<SystemTime> {
    let secs = unix_secs?;
    if secs < 0 {
        return None;
    }
    Some(UNIX_EPOCH + Duration::from_secs(secs as u64))
}

struct Credentials {
    access_token: String,
    expires_at: Option<i64>,
    source: CredentialSource,
}

#[derive(Clone, Debug)]
enum CredentialSource {
    Windows(PathBuf),
    Wsl { distro: String },
}

fn read_first_credentials() -> Option<Credentials> {
    if let Some(creds) = read_windows_credentials() {
        return Some(creds);
    }

    for distro in list_wsl_distros() {
        if let Some(creds) = read_wsl_credentials(&distro) {
            return Some(creds);
        }
    }

    None
}

fn read_windows_credentials() -> Option<Credentials> {
    let CredentialSource::Windows(cred_path) = windows_credential_source()? else {
        return None;
    };
    let content = match std::fs::read_to_string(&cred_path) {
        Ok(content) => content,
        Err(error) => {
            if diagnose::is_enabled() {
                diagnose::log_error(
                    &format!(
                        "unable to read Windows credentials at {}",
                        cred_path.display()
                    ),
                    error,
                );
            }
            return None;
        }
    };
    parse_credentials(&content, CredentialSource::Windows(cred_path))
}

fn read_credentials_from_source(source: &CredentialSource) -> Option<Credentials> {
    match source {
        CredentialSource::Windows(path) => {
            let content = std::fs::read_to_string(path).ok()?;
            parse_credentials(&content, source.clone())
        }
        CredentialSource::Wsl { distro } => read_wsl_credentials(distro),
    }
}

fn codex_auth_path() -> Option<PathBuf> {
    if let Some(codex_home) = std::env::var_os("CODEX_HOME").map(PathBuf::from) {
        return Some(codex_home.join("auth.json"));
    }

    Some(dirs::home_dir()?.join(".codex").join("auth.json"))
}

fn read_codex_credentials() -> Option<CodexTokenData> {
    let auth_path = codex_auth_path()?;
    let content = match std::fs::read_to_string(&auth_path) {
        Ok(content) => content,
        Err(error) => {
            diagnose::log_error(
                &format!(
                    "unable to read Codex credentials at {}",
                    auth_path.display()
                ),
                error,
            );
            return None;
        }
    };

    let auth: CodexAuthFile = serde_json::from_str(&content).ok()?;
    auth.tokens.filter(|tokens| !tokens.access_token.is_empty())
}

fn read_antigravity_credentials() -> Option<AntigravityTokenData> {
    let content = read_windows_generic_credential(ANTIGRAVITY_CREDENTIAL_TARGET)?;
    let auth: AntigravityAuthFile = serde_json::from_str(&content).ok()?;
    if auth.token.access_token.is_empty() {
        None
    } else {
        Some(auth.token)
    }
}

fn read_windows_generic_credential(target: &str) -> Option<String> {
    const CRED_TYPE_GENERIC: u32 = 1;

    let mut target_wide: Vec<u16> = target.encode_utf16().chain(std::iter::once(0)).collect();
    let mut credential: *mut CredentialW = std::ptr::null_mut();

    let ok = unsafe {
        CredReadW(
            target_wide.as_mut_ptr(),
            CRED_TYPE_GENERIC,
            0,
            &mut credential,
        )
    };

    if ok == 0 || credential.is_null() {
        diagnose::log(format!(
            "unable to read Windows generic credential target {target}"
        ));
        return None;
    }

    let result = unsafe {
        let cred = &*credential;
        if cred.credential_blob_size == 0 || cred.credential_blob.is_null() {
            CredFree(credential as *mut c_void);
            return None;
        }
        let bytes =
            std::slice::from_raw_parts(cred.credential_blob, cred.credential_blob_size as usize);
        let text = String::from_utf8(bytes.to_vec()).ok();
        CredFree(credential as *mut c_void);
        text
    };

    result
}

fn read_wsl_credentials(distro: &str) -> Option<Credentials> {
    let output = run_with_timeout(
        Command::new("wsl.exe")
            .arg("-d")
            .arg(distro)
            .arg("--")
            .arg("sh")
            .arg("-lc")
            .arg("cat ~/.claude/.credentials.json")
            .creation_flags(CREATE_NO_WINDOW)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null()),
        Duration::from_secs(5),
    )?;

    if !output.status.success() {
        diagnose::log(format!(
            "WSL credentials probe failed for distro {distro} with status {}",
            output.status
        ));
        return None;
    }

    let content = String::from_utf8(output.stdout).ok()?;
    parse_credentials(
        &content,
        CredentialSource::Wsl {
            distro: distro.to_string(),
        },
    )
}

fn parse_credentials(content: &str, source: CredentialSource) -> Option<Credentials> {
    let json: serde_json::Value = serde_json::from_str(content).ok()?;

    let oauth = json.get("claudeAiOauth")?;
    let access_token = oauth
        .get("accessToken")
        .and_then(|v| v.as_str())?
        .to_string();
    let expires_at = oauth.get("expiresAt").and_then(|v| v.as_i64());

    Some(Credentials {
        access_token,
        expires_at,
        source,
    })
}

fn read_next_credentials_after(source: &CredentialSource) -> Option<Credentials> {
    match source {
        CredentialSource::Windows(_) => {
            for distro in list_wsl_distros() {
                if let Some(creds) = read_wsl_credentials(&distro) {
                    return Some(creds);
                }
            }
        }
        CredentialSource::Wsl { distro } => {
            let mut past_current = false;
            for candidate_distro in list_wsl_distros() {
                if !past_current {
                    past_current = candidate_distro == *distro;
                    continue;
                }
                if let Some(creds) = read_wsl_credentials(&candidate_distro) {
                    return Some(creds);
                }
            }
        }
    }

    None
}

fn list_wsl_distros() -> Vec<String> {
    let output = match run_with_timeout(
        Command::new("wsl.exe")
            .args(["-l", "-q"])
            .creation_flags(CREATE_NO_WINDOW)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null()),
        Duration::from_secs(5),
    ) {
        Some(output) if output.status.success() => output,
        _ => {
            diagnose::log("unable to enumerate WSL distros");
            return Vec::new();
        }
    };

    let stdout = decode_wsl_text(&output.stdout);
    stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn decode_wsl_text(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return String::new();
    }

    if let Some(decoded) = decode_utf16le(bytes) {
        return decoded;
    }

    String::from_utf8_lossy(bytes).into_owned()
}

fn decode_utf16le(bytes: &[u8]) -> Option<String> {
    if bytes.len() < 2 || bytes.len() % 2 != 0 {
        return None;
    }

    let body = if bytes.starts_with(&[0xFF, 0xFE]) {
        &bytes[2..]
    } else if looks_like_utf16le(bytes) {
        bytes
    } else {
        return None;
    };

    let units: Vec<u16> = body
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();

    Some(String::from_utf16_lossy(&units))
}

fn looks_like_utf16le(bytes: &[u8]) -> bool {
    let sample_len = bytes.len().min(128);
    let units = sample_len / 2;
    if units == 0 {
        return false;
    }

    let nul_high_bytes = bytes[..sample_len]
        .chunks_exact(2)
        .filter(|chunk| chunk[1] == 0)
        .count();

    nul_high_bytes * 2 >= units
}

fn is_token_expired(expires_at: Option<i64>) -> bool {
    let Some(exp) = expires_at else { return false };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;
    now >= exp
}

/// Parse an ISO 8601 timestamp string into a SystemTime.
fn parse_iso8601(s: Option<&str>) -> Option<SystemTime> {
    let s = s?;
    // Strip timezone offset to get "YYYY-MM-DDTHH:MM:SS" or with fractional seconds
    // The API returns formats like "2026-03-05T08:00:00.321598+00:00"
    let datetime_part = s.split('+').next().unwrap_or(s);
    let datetime_part = datetime_part.split('Z').next().unwrap_or(datetime_part);

    // Try parsing with and without fractional seconds
    let formats = ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S"];
    for fmt in &formats {
        if let Ok(secs) = parse_datetime_to_unix(datetime_part, fmt) {
            return Some(UNIX_EPOCH + Duration::from_secs(secs));
        }
    }
    None
}

/// Minimal datetime parser — avoids pulling in chrono/time crates.
fn parse_datetime_to_unix(s: &str, _fmt: &str) -> Result<u64, ()> {
    // Extract date and time parts from "YYYY-MM-DDTHH:MM:SS[.frac]"
    let (date_str, time_str) = s.split_once('T').ok_or(())?;
    let date_parts: Vec<&str> = date_str.split('-').collect();
    if date_parts.len() != 3 {
        return Err(());
    }

    let year: u64 = date_parts[0].parse().map_err(|_| ())?;
    let month: u64 = date_parts[1].parse().map_err(|_| ())?;
    let day: u64 = date_parts[2].parse().map_err(|_| ())?;

    // Strip fractional seconds
    let time_base = time_str.split('.').next().unwrap_or(time_str);
    let time_parts: Vec<&str> = time_base.split(':').collect();
    if time_parts.len() != 3 {
        return Err(());
    }

    let hour: u64 = time_parts[0].parse().map_err(|_| ())?;
    let min: u64 = time_parts[1].parse().map_err(|_| ())?;
    let sec: u64 = time_parts[2].parse().map_err(|_| ())?;

    // Days from year (using a simplified calculation for dates after 1970)
    let mut days: u64 = 0;
    for y in 1970..year {
        days += if is_leap(y) { 366 } else { 365 };
    }

    let month_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    for m in 1..month {
        days += month_days[m as usize];
        if m == 2 && is_leap(year) {
            days += 1;
        }
    }
    days += day - 1;

    Ok(days * 86400 + hour * 3600 + min * 60 + sec)
}

fn is_leap(y: u64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

/// Format a usage section as "X% · Yh" style text
pub fn format_line(section: &UsageSection, strings: Strings) -> String {
    let pct = format!("{:.0}%", section.percentage);
    let cd = format_countdown(section.resets_at, strings);
    if cd.is_empty() {
        pct
    } else {
        format!("{pct} \u{00b7} {cd}")
    }
}

fn format_countdown(resets_at: Option<SystemTime>, strings: Strings) -> String {
    let reset = match resets_at {
        Some(t) => t,
        None => return String::new(),
    };

    let remaining = match reset.duration_since(SystemTime::now()) {
        Ok(d) => d,
        Err(_) => return strings.now.to_string(),
    };

    format_countdown_from_secs(remaining.as_secs(), strings)
}

/// Calculate how long until the display text would change
pub fn time_until_display_change(resets_at: Option<SystemTime>) -> Option<Duration> {
    let reset = resets_at?;
    let remaining = reset.duration_since(SystemTime::now()).ok()?;
    Some(time_until_display_change_from_secs(remaining.as_secs()))
}

fn format_countdown_from_secs(total_secs: u64, strings: Strings) -> String {
    let total_mins = total_secs / 60;
    let total_hours = total_secs / 3600;
    let total_days = total_secs / 86400;

    if total_days >= 1 {
        format!("{total_days}{}", strings.day_suffix)
    } else if total_hours >= 1 {
        let remaining_mins = (total_secs % 3600) / 60;
        format!(
            "{total_hours}{}{remaining_mins}{}",
            strings.hour_suffix, strings.minute_suffix
        )
    } else if total_mins >= 1 {
        format!("{total_mins}{}", strings.minute_suffix)
    } else {
        format!("{total_secs}{}", strings.second_suffix)
    }
}

fn time_until_display_change_from_secs(total_secs: u64) -> Duration {
    let total_mins = total_secs / 60;
    let total_hours = total_secs / 3600;
    let total_days = total_secs / 86400;

    let current_bucket_start = if total_days >= 1 {
        total_days * 86400
    } else if total_hours >= 1 {
        total_hours * 3600
    } else if total_mins >= 1 {
        total_mins * 60
    } else {
        total_secs
    };

    Duration::from_secs(total_secs.saturating_sub(current_bucket_start) + 1)
}

/// Returns true if either section has reached "now" (reset time has passed).
pub fn is_past_reset(data: &UsageData) -> bool {
    let now = SystemTime::now();
    let past = |s: &UsageSection| matches!(s.resets_at, Some(t) if now.duration_since(t).is_ok());
    past(&data.session) || past(&data.weekly)
}

pub fn app_is_past_reset(data: &AppUsageData) -> bool {
    data.claude_code.as_ref().is_some_and(is_past_reset)
        || data.codex.as_ref().is_some_and(is_past_reset)
        || data.antigravity.as_ref().is_some_and(is_past_reset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codex_window_duration_controls_row_and_lock() {
        let now = SystemTime::now();
        assert!(codex_reset_is_sliding(
            Some(now + CODEX_WEEKLY_ANCHORED_MAX_REMAINING + Duration::from_secs(30)),
            CODEX_WEEKLY_ANCHORED_MAX_REMAINING,
        ));
        assert!(!codex_reset_is_sliding(
            Some(now + CODEX_WEEKLY_ANCHORED_MAX_REMAINING - Duration::from_secs(30)),
            CODEX_WEEKLY_ANCHORED_MAX_REMAINING,
        ));

        let response: CodexUsageResponse = serde_json::from_str(
            r#"{"rate_limit":{"primary_window":{"used_percent":4,"limit_window_seconds":604800,"reset_at":2000000000},"secondary_window":null}}"#,
        )
        .unwrap();

        let (usage, five_hour_limit_unavailable) = codex_usage_from_response(response);

        assert_eq!(usage.session.percentage, 0.0);
        assert_eq!(usage.weekly.percentage, 4.0);
        assert!(five_hour_limit_unavailable);
        assert!(should_trigger_codex(false, false, false, true, true));
        assert!(!should_trigger_codex(false, true, false, true, true));

        let restored: CodexUsageResponse = serde_json::from_str(
            r#"{"rate_limit":{"primary_window":{"used_percent":1,"limit_window_seconds":18000,"reset_at":2000000000},"secondary_window":null}}"#,
        )
        .unwrap();
        let (usage, five_hour_limit_unavailable) = codex_usage_from_response(restored);

        assert_eq!(usage.session.percentage, 1.0);
        assert!(!five_hour_limit_unavailable);
        assert!(should_trigger_codex(false, false, true, false, true));
    }

    fn usage_with_session_percent(percentage: f64) -> UsageData {
        UsageData {
            session: UsageSection {
                percentage,
                resets_at: None,
            },
            weekly: UsageSection::default(),
        }
    }

    #[test]
    fn empty_codex_window_overrides_stale_anchor() {
        assert!(should_trigger_codex(false, true, true, true, true));
        assert!(!should_trigger_codex(false, true, false, true, true));
    }

    #[test]
    fn claude_failure_does_not_block_codex_when_both_are_enabled() {
        let data = poll_with(
            true,
            true,
            false,
            || Err(PollError::AuthRequired),
            || Ok(usage_with_session_percent(42.0)),
            || unreachable!("antigravity is disabled"),
        )
        .expect("codex data should keep the poll successful");

        assert!(data.claude_code.is_none());
        assert_eq!(data.codex.unwrap().session.percentage, 42.0);
    }

    #[test]
    fn codex_failure_does_not_block_claude_when_both_are_enabled() {
        let data = poll_with(
            true,
            true,
            false,
            || Ok(usage_with_session_percent(64.0)),
            || Err(PollError::RequestFailed),
            || unreachable!("antigravity is disabled"),
        )
        .expect("claude data should keep the poll successful");

        assert_eq!(data.claude_code.unwrap().session.percentage, 64.0);
        assert!(data.codex.is_none());
    }

    #[test]
    fn returns_first_error_when_no_enabled_provider_succeeds() {
        let error = poll_with(
            true,
            true,
            true,
            || Err(PollError::AuthRequired),
            || Err(PollError::RequestFailed),
            || Err(PollError::NoCredentials),
        )
        .expect_err("all-provider failure should return an error");

        assert_eq!(error, PollError::AuthRequired);
    }

    #[test]
    fn antigravity_failure_does_not_block_codex_when_both_are_enabled() {
        let data = poll_with(
            false,
            true,
            true,
            || unreachable!("claude code is disabled"),
            || Ok(usage_with_session_percent(42.0)),
            || Err(PollError::NoCredentials),
        )
        .expect("codex data should keep the poll successful");

        assert!(data.antigravity.is_none());
        assert_eq!(data.codex.unwrap().session.percentage, 42.0);
    }

    #[test]
    fn antigravity_summary_prefers_gemini_group() {
        let response: AntigravityQuotaSummaryResponse = serde_json::from_str(
            r#"{
                "groups": [
                    {
                        "displayName": "Claude and GPT models",
                        "buckets": [
                            {
                                "bucketId": "3p-weekly",
                                "window": "weekly",
                                "resetTime": "2026-06-20T18:32:02Z",
                                "remainingFraction": 1
                            },
                            {
                                "bucketId": "3p-5h",
                                "window": "5h",
                                "resetTime": "2026-06-13T23:32:02Z",
                                "remainingFraction": 1
                            }
                        ]
                    },
                    {
                        "displayName": "Gemini Models",
                        "description": "Models within this group: Gemini Flash, Gemini Pro",
                        "buckets": [
                            {
                                "bucketId": "gemini-weekly",
                                "displayName": "Weekly Limit",
                                "window": "weekly",
                                "resetTime": "2026-06-20T17:08:54Z",
                                "remainingFraction": 0.99304295
                            },
                            {
                                "bucketId": "gemini-5h",
                                "displayName": "Five Hour Limit",
                                "window": "5h",
                                "resetTime": "2026-06-13T22:08:54Z",
                                "remainingFraction": 0.9582575
                            }
                        ]
                    }
                ]
            }"#,
        )
        .expect("summary response should deserialize");

        let usage =
            antigravity_usage_from_summary(response).expect("Gemini quota should be selected");

        assert!((usage.weekly.percentage - 0.695705).abs() < 0.000001);
        assert!((usage.session.percentage - 4.17425).abs() < 0.000001);
        assert!(usage.weekly.resets_at.is_some());
        assert!(usage.session.resets_at.is_some());
    }
}
