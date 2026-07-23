use crate::agentic::core::{Message, MessageContent, MessageRole};
use crate::util::errors::{BitFunError, BitFunResult};
use base64::Engine;
use bitfun_runtime_ports::WorkspaceServices;
use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};

pub const ROUND_REPLAY_CHECKPOINT_VERSION: u32 = 2;
pub const ROUND_REPLAY_START_CONTEXT_KEY: &str = "eval_round_replay_start";
pub const ROUND_REPLAY_ARTIFACT_SESSION_CONTEXT_KEY: &str = "eval_round_replay_artifact_session_id";
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
    #[serde(default)]
    pub artifact_session_id: Option<String>,
    pub agent_type: String,
    pub original_user_input: String,
    pub primary_model_id: String,
    #[serde(default)]
    pub session_max_context_tokens: Option<usize>,
    pub messages: Vec<Message>,
    pub workspace: RoundWorkspaceSnapshot,
    #[serde(default)]
    pub session_artifacts: Vec<RoundSessionArtifact>,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct RoundSessionArtifact {
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
        if self.version == 0 || self.version > ROUND_REPLAY_CHECKPOINT_VERSION {
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
        if self
            .artifact_session_id
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(BitFunError::Validation(
                "Round replay checkpoint has an empty artifact session id".to_string(),
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
        for artifact in &self.session_artifacts {
            validate_session_artifact_path(&artifact.relative_path)?;
        }
        Ok(())
    }
}

pub fn checkpoint_recording_enabled() -> bool {
    std::env::var(ROUND_CHECKPOINT_RECORD_ENV)
        .ok()
        .is_some_and(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

pub fn normalize_routed_model_history(messages: &mut [Message], routed_model_id: &str) {
    if !routed_model_id.to_ascii_lowercase().contains("deepseek") {
        return;
    }
    for message in messages {
        if message.role != MessageRole::Assistant {
            continue;
        }
        if let MessageContent::Mixed {
            reasoning_content,
            tool_calls,
            ..
        } = &mut message.content
        {
            if !tool_calls.is_empty() && reasoning_content.is_none() {
                *reasoning_content = Some(String::new());
            }
        }
    }
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

    let decoded_untracked = snapshot
        .untracked_files
        .iter()
        .map(|file| {
            base64::engine::general_purpose::STANDARD
                .decode(&file.contents_base64)
                .map(|contents| (file, contents))
                .map_err(|error| {
                    BitFunError::Validation(format!(
                        "Invalid base64 for checkpoint file {}: {error}",
                        file.relative_path
                    ))
                })
        })
        .collect::<BitFunResult<Vec<_>>>()?;

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
            "git apply --check --binary --whitespace=nowarn .git/bitfun-round-replay.patch",
        )
        .await?;
        run_git(
            services,
            "git apply --binary --whitespace=nowarn .git/bitfun-round-replay.patch",
        )
        .await?;
    }

    for (file, contents) in decoded_untracked {
        let target = Path::new(workspace_root).join(&file.relative_path);
        if let Err(error) = services
            .fs
            .write_file(&target.to_string_lossy(), &contents)
            .await
        {
            let rollback = rollback_workspace_restore(services).await;
            return Err(BitFunError::io(format!(
                "Failed to restore checkpoint file {}: {error}; rollback={rollback}",
                file.relative_path
            )));
        }
    }

    Ok(())
}

pub async fn capture_session_artifacts(
    source_session_dir: &Path,
) -> BitFunResult<Vec<RoundSessionArtifact>> {
    let mut artifacts = Vec::new();
    let mut total_bytes = 0usize;
    for relative_root in ["tool-results", "artifacts/compression-transcripts"] {
        let root = source_session_dir.join(relative_root);
        if !root.is_dir() {
            continue;
        }
        let mut pending = vec![root];
        while let Some(directory) = pending.pop() {
            let mut entries = tokio::fs::read_dir(&directory).await.map_err(|error| {
                BitFunError::io(format!(
                    "Failed to enumerate round replay session artifacts {}: {error}",
                    directory.display()
                ))
            })?;
            while let Some(entry) = entries.next_entry().await.map_err(|error| {
                BitFunError::io(format!("Failed to read session artifact entry: {error}"))
            })? {
                let file_type = entry.file_type().await.map_err(|error| {
                    BitFunError::io(format!("Failed to inspect session artifact: {error}"))
                })?;
                if file_type.is_dir() {
                    pending.push(entry.path());
                    continue;
                }
                if !file_type.is_file() {
                    continue;
                }
                let relative_path = entry
                    .path()
                    .strip_prefix(source_session_dir)
                    .map_err(|error| BitFunError::Validation(error.to_string()))?
                    .to_string_lossy()
                    .replace('\\', "/");
                validate_session_artifact_path(&relative_path)?;
                let contents = tokio::fs::read(entry.path()).await.map_err(|error| {
                    BitFunError::io(format!(
                        "Failed to read session artifact {}: {error}",
                        entry.path().display()
                    ))
                })?;
                total_bytes = total_bytes.saturating_add(contents.len());
                if total_bytes > MAX_CHECKPOINT_BYTES {
                    return Err(checkpoint_size_error(total_bytes));
                }
                artifacts.push(RoundSessionArtifact {
                    relative_path,
                    contents_base64: base64::engine::general_purpose::STANDARD.encode(contents),
                });
            }
        }
    }
    artifacts.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    Ok(artifacts)
}

pub async fn restore_session_artifacts(
    source_session_dir: &Path,
    artifacts: &[RoundSessionArtifact],
) -> BitFunResult<()> {
    let decoded = artifacts
        .iter()
        .map(|artifact| {
            validate_session_artifact_path(&artifact.relative_path)?;
            let contents = base64::engine::general_purpose::STANDARD
                .decode(&artifact.contents_base64)
                .map_err(|error| {
                    BitFunError::Validation(format!(
                        "Invalid base64 for session artifact {}: {error}",
                        artifact.relative_path
                    ))
                })?;
            Ok((artifact, contents))
        })
        .collect::<BitFunResult<Vec<_>>>()?;

    for (artifact, _) in &decoded {
        let target = source_session_dir.join(&artifact.relative_path);
        if target.exists() {
            return Err(BitFunError::Validation(format!(
                "Round replay refuses to overwrite existing session artifact: {}",
                artifact.relative_path
            )));
        }
    }

    let targets = decoded
        .iter()
        .map(|(artifact, _)| source_session_dir.join(&artifact.relative_path))
        .collect::<Vec<_>>();
    for parent in targets.iter().filter_map(|target| target.parent()) {
        tokio::fs::create_dir_all(parent).await.map_err(|error| {
            BitFunError::io(format!(
                "Failed to create session artifact directory {}: {error}",
                parent.display()
            ))
        })?;
    }

    let mut written = Vec::<PathBuf>::new();
    for ((artifact, contents), target) in decoded.into_iter().zip(targets) {
        if let Err(error) = tokio::fs::write(&target, contents).await {
            for path in written.iter().rev() {
                let _ = tokio::fs::remove_file(path).await;
            }
            return Err(BitFunError::io(format!(
                "Failed to restore session artifact {}: {error}",
                artifact.relative_path
            )));
        }
        written.push(target);
    }
    Ok(())
}

async fn rollback_workspace_restore(services: &WorkspaceServices) -> String {
    match run_git(services, "git reset --hard HEAD && git clean -fd").await {
        Ok(_) => "ok".to_string(),
        Err(error) => format!("failed: {error}"),
    }
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

fn validate_session_artifact_path(path: &str) -> BitFunResult<()> {
    validate_relative_path(path)?;
    if !(path.starts_with("tool-results/")
        || path.starts_with("artifacts/compression-transcripts/"))
    {
        return Err(BitFunError::Validation(format!(
            "Unsupported round replay session artifact path: {path}"
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
    use super::{
        capture_session_artifacts, normalize_routed_model_history, restore_session_artifacts,
        validate_relative_path, RoundReplayCheckpoint, ROUND_REPLAY_CHECKPOINT_VERSION,
    };
    use crate::agentic::core::{Message, MessageContent};

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
            artifact_session_id: None,
            agent_type: "agentic".to_string(),
            original_user_input: "task".to_string(),
            primary_model_id: "large".to_string(),
            session_max_context_tokens: Some(128_128),
            messages: Vec::new(),
            workspace: super::RoundWorkspaceSnapshot {
                baseline_commit: "0123456789abcdef".to_string(),
                tracked_patch: String::new(),
                untracked_files: Vec::new(),
            },
            session_artifacts: Vec::new(),
        };
        assert!(checkpoint.validate().is_err());
    }

    #[test]
    fn version_one_checkpoint_remains_readable() {
        let value = serde_json::json!({
            "version": 1,
            "sourceSessionId": "session",
            "sourceTurnId": "turn",
            "sourceTurnIndex": 0,
            "roundIndex": 2,
            "agentType": "agentic",
            "originalUserInput": "task",
            "primaryModelId": "large",
            "messages": [Message::user("task".to_string())],
            "workspace": {
                "baselineCommit": "0123456789abcdef",
                "trackedPatch": "",
                "untrackedFiles": []
            }
        });
        let checkpoint: RoundReplayCheckpoint =
            serde_json::from_value(value).expect("v1 checkpoint should deserialize");
        assert_eq!(checkpoint.version, 1);
        assert!(checkpoint.session_artifacts.is_empty());
        assert_eq!(checkpoint.session_max_context_tokens, None);
        checkpoint
            .validate()
            .expect("v1 checkpoint should validate");
    }

    #[tokio::test]
    async fn session_artifacts_round_trip_only_supported_directories() {
        let source = tempfile::tempdir().expect("source tempdir");
        let tool_results = source.path().join("tool-results");
        tokio::fs::create_dir_all(&tool_results)
            .await
            .expect("create tool-results");
        tokio::fs::write(tool_results.join("result.txt"), b"complete output")
            .await
            .expect("write tool result");
        tokio::fs::create_dir_all(source.path().join("turns"))
            .await
            .expect("create irrelevant directory");
        tokio::fs::write(source.path().join("turns/turn.json"), b"ignored")
            .await
            .expect("write irrelevant file");

        let artifacts = capture_session_artifacts(source.path())
            .await
            .expect("capture artifacts");
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].relative_path, "tool-results/result.txt");

        let destination = tempfile::tempdir().expect("destination tempdir");
        restore_session_artifacts(destination.path(), &artifacts)
            .await
            .expect("restore artifacts");
        assert_eq!(
            tokio::fs::read(destination.path().join("tool-results/result.txt"))
                .await
                .expect("read restored artifact"),
            b"complete output"
        );
        assert!(restore_session_artifacts(destination.path(), &artifacts)
            .await
            .is_err());
    }

    #[test]
    fn deepseek_history_preserves_empty_reasoning_for_tool_rounds() {
        let mut messages = vec![Message::assistant_with_reasoning(
            None,
            String::new(),
            vec![crate::agentic::core::ToolCall {
                tool_id: "call-1".to_string(),
                tool_name: "Read".to_string(),
                arguments: serde_json::json!({"file_path": "/app/file"}),
                raw_arguments: None,
                is_error: false,
                recovered_from_truncation: false,
            }],
        )];
        normalize_routed_model_history(&mut messages, "deepseek-v4-flash");
        let MessageContent::Mixed {
            reasoning_content, ..
        } = &messages[0].content
        else {
            panic!("expected mixed assistant message");
        };
        assert_eq!(reasoning_content.as_deref(), Some(""));
    }
}
