use crate::agentic::core::Message;
use crate::util::errors::{BitFunError, BitFunResult};
use base64::Engine;
use bitfun_runtime_ports::WorkspaceServices;
use serde::{Deserialize, Serialize};
use std::path::{Component, Path};

pub const ROUND_REPLAY_CHECKPOINT_VERSION: u32 = 1;
pub const ROUND_REPLAY_START_CONTEXT_KEY: &str = "eval_round_replay_start";
pub const ROUND_REPLAY_METADATA_KEY: &str = "bitfun_eval_round_replay";
pub const ROUND_CHECKPOINT_RECORD_ENV: &str = "BITFUN_EVAL_RECORD_ROUND_CHECKPOINTS";
pub const ROUND_REPLAY_CHECKPOINT_ENV: &str = "BITFUN_EVAL_REPLAY_CHECKPOINT";

const WORKSPACE_COMMAND_TIMEOUT_MS: u64 = 30_000;
const MAX_CHECKPOINT_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RoundReplayCheckpoint {
    pub version: u32,
    pub source_session_id: String,
    pub source_turn_id: String,
    pub source_turn_index: usize,
    pub round_index: usize,
    pub agent_type: String,
    pub original_user_input: String,
    pub primary_model_id: String,
    pub messages: Vec<Message>,
    pub workspace: RoundWorkspaceSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct RoundWorkspaceSnapshot {
    pub baseline_commit: String,
    pub tracked_patch: String,
    pub untracked_files: Vec<RoundWorkspaceFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct RoundWorkspaceFile {
    pub relative_path: String,
    pub contents_base64: String,
}

impl RoundReplayCheckpoint {
    pub async fn load(path: &Path) -> BitFunResult<Self> {
        let bytes = tokio::fs::read(path).await.map_err(|error| {
            BitFunError::io(format!(
                "Failed to read round replay checkpoint {}: {error}",
                path.display()
            ))
        })?;
        let checkpoint: Self = serde_json::from_slice(&bytes).map_err(|error| {
            BitFunError::Validation(format!(
                "Invalid round replay checkpoint {}: {error}",
                path.display()
            ))
        })?;
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    pub fn validate(&self) -> BitFunResult<()> {
        if self.version != ROUND_REPLAY_CHECKPOINT_VERSION {
            return Err(BitFunError::Validation(format!(
                "Unsupported round replay checkpoint version: {}",
                self.version
            )));
        }
        if self.source_session_id.trim().is_empty()
            || self.source_turn_id.trim().is_empty()
            || self.agent_type.trim().is_empty()
            || self.primary_model_id.trim().is_empty()
        {
            return Err(BitFunError::Validation(
                "Round replay checkpoint is missing required identity fields".to_string(),
            ));
        }
        if self.messages.is_empty() {
            return Err(BitFunError::Validation(
                "Round replay checkpoint contains no model-visible messages".to_string(),
            ));
        }
        validate_commit(&self.workspace.baseline_commit)?;
        for file in &self.workspace.untracked_files {
            validate_relative_path(&file.relative_path)?;
        }
        Ok(())
    }
}

pub fn checkpoint_recording_enabled() -> bool {
    std::env::var(ROUND_CHECKPOINT_RECORD_ENV)
        .ok()
        .is_some_and(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub async fn capture_workspace_snapshot(
    workspace_root: &str,
    services: &WorkspaceServices,
) -> BitFunResult<RoundWorkspaceSnapshot> {
    let baseline_commit = run_git(services, "git rev-parse HEAD").await?;
    let baseline_commit = baseline_commit.trim().to_string();
    validate_commit(&baseline_commit)?;

    let tracked_patch = run_git(
        services,
        "git diff --binary --full-index --no-color HEAD -- .",
    )
    .await?;
    let mut total_bytes = tracked_patch.len();
    if total_bytes > MAX_CHECKPOINT_BYTES {
        return Err(checkpoint_size_error(total_bytes));
    }

    let untracked_output = run_git(services, "git ls-files --others --exclude-standard -z").await?;
    let mut untracked_files = Vec::new();
    for relative_path in untracked_output.split('\0').filter(|path| !path.is_empty()) {
        validate_relative_path(relative_path)?;
        let absolute_path = Path::new(workspace_root).join(relative_path);
        let contents = services
            .fs
            .read_file(&absolute_path.to_string_lossy())
            .await
            .map_err(|error| {
                BitFunError::io(format!(
                    "Failed to capture untracked file {relative_path}: {error}"
                ))
            })?;
        total_bytes = total_bytes.saturating_add(contents.len());
        if total_bytes > MAX_CHECKPOINT_BYTES {
            return Err(checkpoint_size_error(total_bytes));
        }
        untracked_files.push(RoundWorkspaceFile {
            relative_path: relative_path.to_string(),
            contents_base64: base64::engine::general_purpose::STANDARD.encode(contents),
        });
    }

    Ok(RoundWorkspaceSnapshot {
        baseline_commit,
        tracked_patch,
        untracked_files,
    })
}

pub async fn restore_workspace_snapshot(
    workspace_root: &str,
    services: &WorkspaceServices,
    snapshot: &RoundWorkspaceSnapshot,
) -> BitFunResult<()> {
    validate_commit(&snapshot.baseline_commit)?;
    for file in &snapshot.untracked_files {
        validate_relative_path(&file.relative_path)?;
    }

    let current_commit = run_git(services, "git rev-parse HEAD").await?;
    let current_commit = current_commit.trim().to_string();
    if current_commit != snapshot.baseline_commit {
        return Err(BitFunError::Validation(format!(
            "Round replay requires baseline commit {}, but the workspace is at {}",
            snapshot.baseline_commit, current_commit
        )));
    }

    let status = run_git(services, "git status --porcelain --untracked-files=all").await?;
    if !status.trim().is_empty() {
        return Err(BitFunError::Validation(
            "Round replay requires a clean workspace before applying the checkpoint".to_string(),
        ));
    }

    for file in &snapshot.untracked_files {
        let target = Path::new(workspace_root).join(&file.relative_path);
        if services
            .fs
            .exists(&target.to_string_lossy())
            .await
            .map_err(|error| BitFunError::io(error.to_string()))?
        {
            return Err(BitFunError::Validation(format!(
                "Round replay refuses to overwrite existing untracked file: {}",
                file.relative_path
            )));
        }
    }

    if !snapshot.tracked_patch.is_empty() {
        let patch_path = Path::new(workspace_root).join(".git/bitfun-round-replay.patch");
        services
            .fs
            .write_file(
                &patch_path.to_string_lossy(),
                snapshot.tracked_patch.as_bytes(),
            )
            .await
            .map_err(|error| {
                BitFunError::io(format!("Failed to stage round replay patch: {error}"))
            })?;
        run_git(
            services,
            "git apply --binary --whitespace=nowarn .git/bitfun-round-replay.patch",
        )
        .await?;
    }

    for file in &snapshot.untracked_files {
        let contents = base64::engine::general_purpose::STANDARD
            .decode(&file.contents_base64)
            .map_err(|error| {
                BitFunError::Validation(format!(
                    "Invalid base64 for checkpoint file {}: {error}",
                    file.relative_path
                ))
            })?;
        let target = Path::new(workspace_root).join(&file.relative_path);
        services
            .fs
            .write_file(&target.to_string_lossy(), &contents)
            .await
            .map_err(|error| {
                BitFunError::io(format!(
                    "Failed to restore checkpoint file {}: {error}",
                    file.relative_path
                ))
            })?;
    }

    Ok(())
}

async fn run_git(services: &WorkspaceServices, command: &str) -> BitFunResult<String> {
    let (stdout, stderr, exit_code) = services
        .shell
        .exec(command, Some(WORKSPACE_COMMAND_TIMEOUT_MS))
        .await
        .map_err(|error| BitFunError::io(format!("Workspace git command failed: {error}")))?;
    if exit_code != 0 {
        return Err(BitFunError::io(format!(
            "Workspace git command failed with exit code {exit_code}: {}",
            stderr.trim()
        )));
    }
    Ok(stdout)
}

fn validate_commit(commit: &str) -> BitFunResult<()> {
    if commit.len() < 7 || !commit.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(BitFunError::Validation(format!(
            "Invalid checkpoint baseline commit: {commit}"
        )));
    }
    Ok(())
}

fn validate_relative_path(path: &str) -> BitFunResult<()> {
    let parsed = Path::new(path);
    if path.is_empty()
        || parsed.is_absolute()
        || parsed
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(BitFunError::Validation(format!(
            "Unsafe checkpoint workspace path: {path}"
        )));
    }
    Ok(())
}

fn checkpoint_size_error(bytes: usize) -> BitFunError {
    BitFunError::Validation(format!(
        "Round replay checkpoint workspace snapshot exceeds {} bytes (captured {bytes})",
        MAX_CHECKPOINT_BYTES
    ))
}

#[cfg(test)]
mod tests {
    use super::{validate_relative_path, RoundReplayCheckpoint, ROUND_REPLAY_CHECKPOINT_VERSION};

    #[test]
    fn checkpoint_paths_reject_escape_components() {
        assert!(validate_relative_path("src/lib.rs").is_ok());
        assert!(validate_relative_path("../secret").is_err());
        assert!(validate_relative_path("/tmp/secret").is_err());
    }

    #[test]
    fn checkpoint_validation_rejects_empty_messages() {
        let checkpoint = RoundReplayCheckpoint {
            version: ROUND_REPLAY_CHECKPOINT_VERSION,
            source_session_id: "session".to_string(),
            source_turn_id: "turn".to_string(),
            source_turn_index: 0,
            round_index: 2,
            agent_type: "agentic".to_string(),
            original_user_input: "task".to_string(),
            primary_model_id: "large".to_string(),
            messages: Vec::new(),
            workspace: super::RoundWorkspaceSnapshot {
                baseline_commit: "0123456789abcdef".to_string(),
                tracked_patch: String::new(),
                untracked_files: Vec::new(),
            },
        };
        assert!(checkpoint.validate().is_err());
    }
}
