use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const ONE_SHOT_ROUND_MODEL_ROUTE_METADATA_KEY: &str = "bitfun_one_shot_round_model_route";
pub const ONE_SHOT_ROUND_MODEL_ROUTE_MODEL_ENV: &str = "BITFUN_EVAL_SMALL_MODEL";
pub const ONE_SHOT_ROUND_MODEL_ROUTE_ROUND_ENV: &str = "BITFUN_EVAL_SMALL_MODEL_ROUND";

const MAX_MODEL_ID_CHARS: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OneShotRoundModelRoute {
    pub round_number: usize,
    pub model_id: String,
}

impl OneShotRoundModelRoute {
    pub fn new(round_number: usize, model_id: impl Into<String>) -> Result<Self, String> {
        let route = Self {
            round_number,
            model_id: model_id.into().trim().to_string(),
        };
        route.validate()?;
        Ok(route)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.model_id.is_empty() {
            return Err("one-shot round model id must not be empty".to_string());
        }
        if self.model_id.chars().count() > MAX_MODEL_ID_CHARS {
            return Err(format!(
                "one-shot round model id must be at most {MAX_MODEL_ID_CHARS} characters"
            ));
        }
        Ok(())
    }

    pub fn model_id_for_round(&self, round_number: usize) -> Option<&str> {
        (self.round_number == round_number).then_some(self.model_id.as_str())
    }

    pub fn from_metadata(metadata: Option<&Value>) -> Result<Option<Self>, String> {
        let Some(value) = metadata
            .and_then(Value::as_object)
            .and_then(|object| object.get(ONE_SHOT_ROUND_MODEL_ROUTE_METADATA_KEY))
        else {
            return Ok(None);
        };

        let route: Self = serde_json::from_value(value.clone())
            .map_err(|error| format!("invalid one-shot round model route: {error}"))?;
        route.validate()?;
        Ok(Some(route))
    }

    pub fn from_env_values(
        model_id: Option<&str>,
        round_number: Option<&str>,
    ) -> Result<Option<Self>, String> {
        match (model_id, round_number) {
            (Some(model_id), Some(round_number)) => {
                let round_number = round_number.trim().parse::<usize>().map_err(|error| {
                    format!("invalid one-shot round model route round number: {error}")
                })?;
                Self::new(round_number, model_id).map(Some)
            }
            (None, None) => Ok(None),
            _ => Err(format!(
                "{ONE_SHOT_ROUND_MODEL_ROUTE_MODEL_ENV} and {ONE_SHOT_ROUND_MODEL_ROUTE_ROUND_ENV} must be provided together"
            )),
        }
    }

    pub fn to_context_value(&self) -> Result<String, String> {
        serde_json::to_string(self)
            .map_err(|error| format!("failed to serialize one-shot round model route: {error}"))
    }

    pub fn from_context_value(value: &str) -> Result<Self, String> {
        let route: Self = serde_json::from_str(value)
            .map_err(|error| format!("invalid one-shot round model route context: {error}"))?;
        route.validate()?;
        Ok(route)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        OneShotRoundModelRoute, ONE_SHOT_ROUND_MODEL_ROUTE_METADATA_KEY,
        ONE_SHOT_ROUND_MODEL_ROUTE_ROUND_ENV,
    };

    #[test]
    fn selects_only_the_configured_round() {
        let route = OneShotRoundModelRoute::new(3, "small-model").expect("valid route");

        assert_eq!(route.model_id_for_round(2), None);
        assert_eq!(route.model_id_for_round(3), Some("small-model"));
        assert_eq!(route.model_id_for_round(4), None);
    }

    #[test]
    fn metadata_and_context_round_trip() {
        let metadata = serde_json::json!({
            ONE_SHOT_ROUND_MODEL_ROUTE_METADATA_KEY: {
                "roundNumber": 7,
                "modelId": "small-model"
            }
        });

        let route = OneShotRoundModelRoute::from_metadata(Some(&metadata))
            .expect("valid metadata")
            .expect("route present");
        let encoded = route.to_context_value().expect("encode context");

        assert_eq!(
            OneShotRoundModelRoute::from_context_value(&encoded).expect("decode context"),
            route
        );
    }

    #[test]
    fn rejects_empty_model_id() {
        let error = OneShotRoundModelRoute::new(0, "   ").expect_err("empty model id");
        assert!(error.contains("must not be empty"), "{error}");
    }

    #[test]
    fn parses_environment_values() {
        let route = OneShotRoundModelRoute::from_env_values(Some("small-model"), Some("5"))
            .expect("valid environment values")
            .expect("route present");

        assert_eq!(route.round_number, 5);
        assert_eq!(route.model_id, "small-model");
    }

    #[test]
    fn rejects_partial_environment_values() {
        let error = OneShotRoundModelRoute::from_env_values(Some("small-model"), None)
            .expect_err("partial environment values");

        assert!(error.contains(ONE_SHOT_ROUND_MODEL_ROUTE_ROUND_ENV));
    }
}
