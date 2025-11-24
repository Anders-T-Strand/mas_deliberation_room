"""
Model Validator for CrewAI Agents
Verifies that all LLM model names in agents.yaml are valid and accessible
before running the crew.
"""

import os
import yaml
from litellm import completion

# Use exceptions from litellm.exceptions (works across recent versions)
try:
    from litellm.exceptions import (
        AuthenticationError,
        BadRequestError,
        NotFoundError,    # used when model isn't found / not available
        APIError,
        RateLimitError,
        Timeout,
    )
except Exception:  # very old litellm fallback
    AuthenticationError = BadRequestError = NotFoundError = APIError = RateLimitError = Timeout = Exception

REQUIRED_KEYS = {
    "openai/": "OPENAI_API_KEY",
    "anthropic/": "ANTHROPIC_API_KEY",
    "google/": "GOOGLE_API_KEY",
    "mistral/": "MISTRAL_API_KEY",
}

def _check_env_for_model(model: str):
    for prefix, envvar in REQUIRED_KEYS.items():
        if model.startswith(prefix) and not os.getenv(envvar):
            return f"{envvar} not set"
    return None

def validate_models(agent_yaml_path: str = "src/mas_deliberation_room/agents.yaml"):
    print("\n🔍 Validating agent models before crew run...\n")

    with open(agent_yaml_path, "r", encoding="utf-8") as f:
        agents = yaml.safe_load(f)

    errors = []
    for name, cfg in agents.items():
        model = (cfg or {}).get("llm")
        if not model:
            continue

        print(f"🧠 Checking {name}: {model} ... ", end="", flush=True)

        env_issue = _check_env_for_model(model)
        if env_issue:
            print("🚫")
            errors.append((name, model, env_issue))
            continue

        try:
            # Minimal test call — tiny max_tokens to avoid cost
            completion(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
                temperature=0.0,
            )
            print("✅ OK")
        except NotFoundError as e:
            print("❌ Not found")
            errors.append((name, model, f"Model not found/unsupported: {e}"))
        except AuthenticationError as e:
            print("🚫 Auth")
            errors.append((name, model, f"Auth error / bad key: {e}"))
        except (BadRequestError, RateLimitError, Timeout, APIError) as e:
            print("⚠️ API")
            errors.append((name, model, f"Provider error: {e}"))
        except Exception as e:
            print("⚠️ Unknown")
            errors.append((name, model, f"Unexpected: {type(e).__name__}: {e}"))

    if errors:
        print("\n🚨 Model validation failed:")
        for name, model, msg in errors:
            print(f" - {name}: {model} → {msg}")
        raise SystemExit("\n❌ Fix the above model names / API keys, then re-run.\n")

    print("\n✅ All models validated successfully!\n")
