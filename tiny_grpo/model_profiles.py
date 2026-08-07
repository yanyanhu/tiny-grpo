"""Typed model profiles, independent of run length and hardware settings."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class ModelProfile:
    name: str
    model_id: str
    chat_template_mode: str = "default"
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    @property
    def chat_template_kwargs(self) -> dict:
        return chat_template_kwargs(self.chat_template_mode)

    def run_name(self, base_name: str) -> str:
        return base_name if self.name == DEFAULT_MODEL_PROFILE_NAME else f"{base_name}_{self.name}"


def chat_template_kwargs(mode: str) -> dict:
    if mode == "default":
        return {}
    if mode == "thinking":
        return {"enable_thinking": True}
    if mode == "non-thinking":
        return {"enable_thinking": False}
    raise ValueError(f"unknown chat template mode {mode!r}")


SMOLLM2_135M = ModelProfile(
    name="smollm2_135m",
    model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
)

QWEN3_0_6B = ModelProfile(
    name="qwen3_0_6b",
    model_id="Qwen/Qwen3-0.6B",
    chat_template_mode="non-thinking",
)

MODEL_PROFILES = {profile.name: profile for profile in (SMOLLM2_135M, QWEN3_0_6B)}
DEFAULT_MODEL_PROFILE_NAME = SMOLLM2_135M.name


def resolve_model_profile(name: str) -> ModelProfile:
    try:
        return MODEL_PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown model profile {name!r}; expected one of {sorted(MODEL_PROFILES)}") from None
