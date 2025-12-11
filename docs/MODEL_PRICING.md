# Vision Model Pricing & Specifications (December 2025)

This document provides comprehensive pricing and specifications for all supported vision-language models.

## Quick Comparison

| Provider | Model | Input $/1M | Output $/1M | Context | Max Images |
|----------|-------|------------|-------------|---------|------------|
| OpenAI | GPT-5.2 Instant | $1.00 | $4.00 | 200K | 10 |
| OpenAI | GPT-5.2 | $1.75 | $14.00 | 200K | 10 |
| Anthropic | Claude Haiku 4.5 | $1.00 | $5.00 | 200K | 20 |
| Google | Gemini 3.0 Flash | $0.10 | $0.40 | 1M | 3600 |
| Google | Gemini 2.0 Flash | $0.10 | $0.40 | 1M | 3600 |
| Groq | Llama 3.2 11B | $0.02 | $0.02 | 8K | 1 |

**Recommended for real-time agents:** Gemini 3.0 Flash (cheapest), Claude Haiku 4.5 (best quality/speed), GPT-5.2 Instant (best tool calling)

---

## OpenAI

### GPT-5.2 Family (December 2025 - Latest)

| Model ID | Display Name | Input $/1M | Output $/1M | Context | Best For |
|----------|--------------|------------|-------------|---------|----------|
| `gpt-5.2-chat-latest` | GPT-5.2 Instant | $1.00 | $4.00 | 200K | Fast responses, routine queries |
| `gpt-5.2` | GPT-5.2 Thinking | $1.75 | $14.00 | 200K | Complex coding, analysis, planning |
| `gpt-5.2-pro` | GPT-5.2 Pro | $15.00 | $60.00 | 200K | Maximum accuracy, difficult problems |

**Notes:**
- GPT-5.2 supports `reasoning` parameter with levels: `low`, `medium`, `high`, `xhigh`
- 90% discount on cached inputs
- Released December 11, 2025

### GPT-5.1 Family (November 2025)

| Model ID | Display Name | Input $/1M | Output $/1M | Context | Best For |
|----------|--------------|------------|-------------|---------|----------|
| `gpt-5.1` | GPT-5.1 | $1.75 | $14.00 | 200K | Coding, agentic tasks |

**Notes:**
- Snapshot: `gpt-5.1-2025-11-13`
- New "none" reasoning setting for faster responses
- Increased steerability

### GPT-5 Family (August 2025)

| Model ID | Display Name | Input $/1M | Output $/1M | Context | Best For |
|----------|--------------|------------|-------------|---------|----------|
| `gpt-5` | GPT-5 | $2.00 | $8.00 | 200K | General purpose |
| `gpt-5-mini` | GPT-5 Mini | $0.40 | $1.60 | 200K | Balanced cost/performance |
| `gpt-5-nano` | GPT-5 Nano | $0.10 | $0.40 | 200K | High volume, low cost |

### GPT-4.1 Family

| Model ID | Display Name | Input $/1M | Output $/1M | Context | Best For |
|----------|--------------|------------|-------------|---------|----------|
| `gpt-4.1` | GPT-4.1 | $2.00 | $8.00 | 1M | Long context tasks |
| `gpt-4.1-mini` | GPT-4.1 Mini | $0.40 | $1.60 | 1M | Cost-effective long context |
| `gpt-4.1-nano` | GPT-4.1 Nano | $0.10 | $0.40 | 1M | Cheapest long context |

### GPT-4o Family (Legacy)

| Model ID | Display Name | Input $/1M | Output $/1M | Context | Best For |
|----------|--------------|------------|-------------|---------|----------|
| `gpt-4o` | GPT-4o | $2.50 | $10.00 | 128K | Legacy compatibility |
| `gpt-4o-mini` | GPT-4o Mini | $0.15 | $0.60 | 128K | Cheapest OpenAI option |

### Reasoning Models

| Model ID | Display Name | Input $/1M | Output $/1M | Context | Best For |
|----------|--------------|------------|-------------|---------|----------|
| `o3-mini` | o3-mini | $1.15 | $4.40 | 200K | Complex reasoning |

---

## Anthropic

### Claude 4.5 Family (November 2025 - Latest)

| Model ID | Display Name | Input $/1M | Output $/1M | Context | Best For |
|----------|--------------|------------|-------------|---------|----------|
| `claude-opus-4-5-20251101` | Claude Opus 4.5 | $5.00 | $25.00 | 200K | Most intelligent, complex tasks |
| `claude-sonnet-4-5` | Claude Sonnet 4.5 | $3.00 | $15.00 | 200K | Coding, multi-step agents |
| `claude-haiku-4-5` | Claude Haiku 4.5 | $1.00 | $5.00 | 200K | Fast, cost-effective |

**Notes:**
- Opus 4.5: 80.9% on SWE-bench Verified, 66.3% on OSWorld
- Sonnet 4.5: Can maintain focus for 30+ hours on complex tasks
- Haiku 4.5: Similar coding performance to Sonnet 4 at 1/3 cost

### Claude 4 Family (May 2025)

| Model ID | Display Name | Input $/1M | Output $/1M | Context | Best For |
|----------|--------------|------------|-------------|---------|----------|
| `claude-opus-4-20250514` | Claude Opus 4 | $15.00 | $75.00 | 200K | Previous flagship |
| `claude-sonnet-4-20250514` | Claude Sonnet 4 | $3.00 | $15.00 | 200K | Balanced performance |

### Claude 3.5 Family (Legacy)

| Model ID | Display Name | Input $/1M | Output $/1M | Context | Best For |
|----------|--------------|------------|-------------|---------|----------|
| `claude-3-5-sonnet-latest` | Claude 3.5 Sonnet | $3.00 | $15.00 | 200K | Legacy compatibility |
| `claude-3-5-haiku-latest` | Claude 3.5 Haiku | $0.80 | $4.00 | 200K | Budget option |

**All Claude models support:**
- Up to 20 images per request
- Streaming responses
- Tool/function calling

---

## Google

### Gemini 3 Family (November 2025 - Latest)

| Model ID | Display Name | Input $/1M | Output $/1M | Context | Best For |
|----------|--------------|------------|-------------|---------|----------|
| `gemini-3.0-pro` | Gemini 3.0 Pro | $1.25 | $5.00 | 2M | Most capable, complex reasoning |
| `gemini-3.0-flash` | Gemini 3.0 Flash | $0.10 | $0.40 | 1M | Fast, cost-effective |

**Notes:**
- Gemini 3.0 Pro outperformed GPT-5 Pro in 19/20 benchmarks on release
- New `thinking_level` parameter to control reasoning depth
- `media_resolution` parameter for image/video token control
- Deep Think mode available for Ultra subscribers

### Gemini 2.5 Family

| Model ID | Display Name | Input $/1M | Output $/1M | Context | Best For |
|----------|--------------|------------|-------------|---------|----------|
| `gemini-2.5-pro` | Gemini 2.5 Pro | $1.25 | $10.00 | 1M | Complex reasoning, coding |
| `gemini-2.5-flash` | Gemini 2.5 Flash | $0.15 | $0.60 | 1M | Best price-performance |
| `gemini-2.5-flash-lite` | Gemini 2.5 Flash Lite | $0.075 | $0.30 | 1M | Massive scale, high throughput |

### Gemini 2.0 Family

| Model ID | Display Name | Input $/1M | Output $/1M | Context | Best For |
|----------|--------------|------------|-------------|---------|----------|
| `gemini-2.0-flash` | Gemini 2.0 Flash | $0.10 | $0.40 | 1M | Reliable, well-tested |

### Gemini 1.5 Family (Legacy)

| Model ID | Display Name | Input $/1M | Output $/1M | Context | Best For |
|----------|--------------|------------|-------------|---------|----------|
| `gemini-1.5-pro` | Gemini 1.5 Pro | $1.25 | $5.00 | 2M | Legacy, long context |
| `gemini-1.5-flash` | Gemini 1.5 Flash | $0.075 | $0.30 | 1M | Legacy, budget |

**All Gemini models support:**
- Up to 3600 images per request
- Native video understanding
- Tool/function calling
- Massive context windows (1-2M tokens)

---

## Groq

| Model ID | Display Name | Input $/1M | Output $/1M | Context | Best For |
|----------|--------------|------------|-------------|---------|----------|
| `llama-3.2-90b-vision-preview` | Llama 3.2 90B Vision | $0.90 | $0.90 | 8K | Fastest inference |
| `llama-3.2-11b-vision-preview` | Llama 3.2 11B Vision | $0.02 | $0.02 | 8K | Ultra-low cost |

**Notes:**
- Groq provides the fastest inference speeds
- Limited to 1 image per request
- 8K context window

---

## Fireworks

| Model ID | Display Name | Input $/1M | Output $/1M | Context | Best For |
|----------|--------------|------------|-------------|---------|----------|
| `firellava-13b` | FireLLaVA 13B | $0.20 | $0.20 | 4K | Open model, good value |

---

## Together

| Model ID | Display Name | Input $/1M | Output $/1M | Context | Best For |
|----------|--------------|------------|-------------|---------|----------|
| `llama-3.2-11b-vision` | Llama 3.2 11B Vision | $0.18 | $0.18 | 8K | Open model option |
| `llama-3.2-90b-vision` | Llama 3.2 90B Vision | $1.20 | $1.20 | 8K | Higher quality open model |

---

## Modal (Self-Hosted)

Modal allows you to deploy any vision model serverlessly. Pricing depends on GPU usage.

| Model Profile | Estimated $/1M tokens | GPU | Best For |
|---------------|----------------------|-----|----------|
| `llama-3.2-90b-vision` | ~$0.90 | A100 | High quality, self-hosted |
| `llama-3.2-11b-vision` | ~$0.20 | A10G | Cost-effective self-hosted |
| `qwen2-vl-72b` | ~$0.80 | A100 | Video understanding |
| `pixtral-12b` | ~$0.20 | A10G | Fast, efficient |

**Setup:**
```bash
export MODAL_ENDPOINT_URL="https://your-workspace--vision-model.modal.run"
export MODAL_MODEL_ID="llama-3.2-90b-vision"
```

---

## Cost Optimization Tips

### For Real-Time Agents (< 1 second latency needed)
1. **Cheapest:** Gemini 3.0 Flash ($0.10/$0.40)
2. **Best quality:** Claude Haiku 4.5 ($1.00/$5.00)
3. **Best tool calling:** GPT-5.2 Instant ($1.00/$4.00)

### For Batch Processing (cost matters most)
1. **Cheapest:** Groq Llama 3.2 11B ($0.02/$0.02)
2. **Best value:** Gemini 2.5 Flash Lite ($0.075/$0.30)
3. **Good balance:** GPT-5-nano ($0.10/$0.40)

### For Complex Analysis (quality matters most)
1. **Best overall:** GPT-5.2 Pro or Gemini 3.0 Pro
2. **Best for coding:** Claude Opus 4.5
3. **Best for long context:** Gemini (1-2M tokens)

### Image Token Estimation
- **Low detail:** ~85 tokens per 512x512 image
- **High detail:** ~765 tokens per 512x512 image
- **Gemini:** Use `media_resolution` parameter to control

---

## Version History

| Date | Updates |
|------|---------|
| Dec 11, 2025 | Added GPT-5.2 family |
| Nov 24, 2025 | Added Claude 4.5 family |
| Nov 18, 2025 | Added Gemini 3.0 family |
| Nov 13, 2025 | Added GPT-5.1 |
| Aug 7, 2025 | Added GPT-5 family |

---

## Links

- [OpenAI Pricing](https://openai.com/pricing)
- [Anthropic Pricing](https://www.anthropic.com/pricing)
- [Google AI Pricing](https://ai.google.dev/pricing)
- [Groq Pricing](https://groq.com/pricing)
- [Fireworks Pricing](https://fireworks.ai/pricing)
- [Together Pricing](https://www.together.ai/pricing)
- [Modal Pricing](https://modal.com/pricing)
