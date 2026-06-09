"""
Standardized backend registry data
- order: numeric string used for sorting
- id: lowercase, no spaces/specials (used in URLs & images)
- topics / duration: written in display-ready Title Case so the FE can
  render them as-is (no client-side casing transforms).
"""

# Categories: List[Dict]
CATEGORIES = [
    {"order": "01", "id": "ai", "name": "AI", "description": "Hands-on labs for core AI/LLM product skills."},
    {"order": "02", "id": "ai-security", "name": "AI Security", "description": "Hands-on labs for AI Security product skills."},
]

# Subcategories: Dict[category_id] -> List[Dict]
SUBCATEGORIES = {
    "ai": [
        {
            "order": "01",
            "id": "foundations",
            "name": "Foundations",
            "description": "Tokens, context windows, embeddings, and sampling.",
            "duration": "15-20 Minutes",
            "topics": ["Systems", "LLMs"],
        },
        {
            "order": "02",
            "id": "foundations-2",
            "name": "Foundations 2",
            "description": "Advanced parameter interactions and complex sampling strategies.",
            "duration": "20-30 Minutes",
            "topics": ["Hyperparameters", "Interactions"],
        },
    ],
    "ai-security": [
        {
            "order": "01",
            "id": "ai-app-sec",
            "name": "AI App Sec",
            "description": "Adversarial prompting fundamentals — attack vectors, mitigations, and hands-on practice against LLM apps.",
            "duration": "15-25 Minutes",
            "topics": ["AI Security", "Prompting"],
        },
        {
            "order": "02",
            "id": "ai-data-exfiltration",
            "name": "AI Data Exfiltration",
            "description": "Extract sensitive information from LLM apps via prompt manipulation and indirect channels.",
            "duration": "15-25 Minutes",
            "topics": ["AI Security", "Data Exfiltration"],
        },
        {
            "order": "03",
            "id": "ai-model-extraction",
            "name": "AI Model Extraction",
            "description": "Steal a model's capabilities through API access, distillation, and similar IP-theft attacks — plus the watermarking defenses against them.",
            "duration": "15-25 Minutes",
            "topics": ["AI Security", "Model Extraction", "Watermarking"],
        },
        {
            "order": "04",
            "id": "image-adversarial-attacks",
            "name": "Image Adversarial Attacks",
            "description": "White-box pixel perturbation attacks (FGSM, PGD, CW, DeepFool, SmoothFool) on ImageNet classifiers.",
            "duration": "25-40 Minutes",
            "topics": ["AI Security", "Adversarial ML", "Computer Vision"],
        },
        {
            "order": "05",
            "id": "vision-llm-attacks",
            "name": "Vision LLM Attacks",
            "description": "Adversarial attacks targeting vision-language models.",
            "duration": "20-30 Minutes",
            "topics": ["AI Security", "Vision LLM"],
        },
        {
            "order": "06",
            "id": "ai-supply-chain-attacks",
            "name": "AI Supply Chain Attacks",
            "description": "Poisoned checkpoints and poisoned datasets — train-time and load-time attacks that implant backdoors in the model itself.",
            "duration": "20-40 Minutes",
            "topics": ["AI Security", "Supply Chain", "Backdoor"],
        },
    ],
}

# Apps: Dict[category_id] -> Dict[subcategory_id] -> List[Dict]
# NOTE: Uses category/subcategory **id** keys (not order). Each app has:
#   - order: numeric string for sorting within the subcategory
#   - id: slug used for URLs and images
APPS = {
    "ai": {  # AI
        "foundations": [  # Foundations
            {
                "order": "01",
                "id": "temperature-101",
                "appId": "temperature-101",
                "name": "Temperature 101: From Logic to Chaos",
                "description": "Master temperature via an interactive tour; understand randomness vs. creativity/logic.",
                "topics": ["Temperature", "Sampling"],
                "duration": "15-20 Minutes",
            },
            {
                "order": "02",
                "id": "topk-101",
                "appId": "topk-101",
                "name": "Top-K 101: Taming the Tail",
                "description": "Understand how Top-K trims unlikely tokens to prevent incoherence; greedy decoding vs. wide sampling.",
                "topics": ["Top-K", "Decoding"],
                "duration": "15-20 Minutes",
            },
            {
                "order": "03",
                "id": "topp-101",
                "appId": "topp-101",
                "name": "Top-P 101: Nucleus Sampling",
                "description": "Learn Top-P; compare with Top-K & Temperature; when to use for natural-sounding text.",
                "topics": ["Top-P", "Nucleus"],
                "duration": "15-20 Minutes",
            },
            {
                "order": "04",
                "id": "presence-penalty-101",
                "appId": "presence-penalty-101",
                "name": "Presence Penalty: Preventing Repetition",
                "description": "How presence penalty discourages repeating topics; control breadth of conversation.",
                "topics": ["Penalties", "Repetition"],
                "duration": "15-20 Minutes",
            },
            {
                "order": "05",
                "id": "frequency-penalty-101",
                "appId": "frequency-penalty-101",
                "name": "Frequency Penalty: Word Repetition",
                "description": "Discourage repeating words; avoid stuttering or loops.",
                "topics": ["Penalties", "Frequency"],
                "duration": "15-20 Minutes",
            },
        ],
        "foundations-2": [  # Foundations 2
            {
                "order": "01",
                "id": "temp-vs-topk",
                "appId": "temp-vs-topk",
                "name": "Temp vs Top-K: The Power Struggle",
                "description": "See interactions/overrides; why Top-K=1 makes Temperature irrelevant.",
                "topics": ["Temperature", "Top-K"],
                "duration": "20-30 Minutes",
            },
            {
                "order": "02",
                "id": "temp-vs-topp",
                "appId": "temp-vs-topp",
                "name": "Temp vs Top-P: The Safety Net",
                "description": "Low Top-P can tame high Temperature; hard cut vs. cumulative sum trade-offs.",
                "topics": ["Temperature", "Top-P"],
                "duration": "20-30 Minutes",
            },
            {
                "order": "03",
                "id": "topk-vs-topp",
                "appId": "topk-vs-topp",
                "name": "Top-K vs Top-P: The Filter Funnel",
                "description": "Double filter strategy; K acts coarse, P acts fine; order of operations.",
                "topics": ["Top-K", "Top-P"],
                "duration": "20-30 Minutes",
            },
            {
                "order": "04",
                "id": "creativity-cocktail",
                "appId": "creativity-cocktail",
                "name": "The Creativity Cocktail",
                "description": "Explore Temperature + Frequency + Presence penalties; break loops without losing coherence.",
                "topics": ["Temperature", "Penalties"],
                "duration": "20-30 Minutes",
            },
        ],
    },
    "ai-security": {  # AI Security
        "ai-app-sec": [  # AI App Sec
            {
                "order": "01",
                "id": "crdlr",
                "appId": "crdlr",
                "name": "CRDLR: Adversarial Prompting Fundamentals",
                "description": "Fundamentals of adversarial prompting; attack vectors & mitigations; hands-on practice.",
                "topics": ["Security", "Prompting"],
                "duration": "15-25 Minutes",
            },
        ],
        "ai-data-exfiltration": [  # AI Data Exfiltration
            {
                "order": "01",
                "id": "echoleak",
                "appId": "echoleak",
                "name": "Echoleak: Information Disclosure",
                "description": "Master information extraction from multiple sources; understand data leakage vectors.",
                "topics": ["Security", "Information Disclosure"],
                "duration": "15-25 Minutes",
            },
        ],
        "ai-model-extraction": [  # AI Model Extraction
            {
                "order": "01",
                "id": "modeltheftimages",
                "appId": "modeltheftimages",
                "name": "Model Theft: Image Generation & Watermarking",
                "description": "Explore model theft through diffusion image generation. See how models can be stolen via API access and how watermarking defends against it.",
                "topics": ["Security", "Model Theft", "Diffusion", "Watermarking"],
                "duration": "15-25 Minutes",
            },
        ],
        "image-adversarial-attacks": [  # Image Adversarial Attacks
            {
                "order": "01",
                "id": "imgadv-fgsm",
                "appId": "imgadv-fgsm",
                "name": "FGSM: Fast Gradient Sign Method",
                "description": "Craft an adversarial image in a single signed-gradient step bounded by ε. The fastest white-box attack — see how one step flips an ImageNet classifier.",
                "topics": ["Security", "Adversarial", "FGSM", "Vision"],
                "duration": "15-20 Minutes",
            },
            {
                "order": "02",
                "id": "imgadv-pgd",
                "appId": "imgadv-pgd",
                "name": "PGD: Projected Gradient Descent",
                "description": "Iterate FGSM, projecting back into the ε-ball each step. The strongest first-order L∞ attack — watch the perturbation refine over epochs.",
                "topics": ["Security", "Adversarial", "PGD", "Vision"],
                "duration": "20-30 Minutes",
            },
            {
                "order": "03",
                "id": "imgadv-cw",
                "appId": "imgadv-cw",
                "name": "Carlini–Wagner (L2)",
                "description": "Minimize ‖δ‖₂ + c·f(x+δ) with Adam in tanh space to find a near-imperceptible perturbation. The classic optimization attack.",
                "topics": ["Security", "Adversarial", "Carlini-Wagner", "Vision"],
                "duration": "25-35 Minutes",
            },
            {
                "order": "04",
                "id": "imgadv-deepfool",
                "appId": "imgadv-deepfool",
                "name": "DeepFool",
                "description": "Push the image across the nearest linearized decision boundary, iteration by iteration, to find the minimal perturbation that fools the model.",
                "topics": ["Security", "Adversarial", "DeepFool", "Vision"],
                "duration": "20-30 Minutes",
            },
            {
                "order": "05",
                "id": "imgadv-smoothfool",
                "appId": "imgadv-smoothfool",
                "name": "SmoothFool",
                "description": "A DeepFool variant that low-pass filters the perturbation, producing smooth, structured noise. Tune the smoothing σ and compare.",
                "topics": ["Security", "Adversarial", "SmoothFool", "Vision"],
                "duration": "20-30 Minutes",
            },
            {
                "order": "06",
                "id": "imgadv-playground",
                "appId": "imgadv-playground",
                "name": "Adversarial Playground",
                "description": "Pick any attack (FGSM, PGD, CW, DeepFool, SmoothFool), tune every hyperparameter, and compare their perturbations on the same image side by side.",
                "topics": ["Security", "Adversarial", "Playground", "Vision"],
                "duration": "20-40 Minutes",
            },
        ],
        "vision-llm-attacks": [  # Vision LLM Attacks
            {
                "order": "01",
                "id": "figstep",
                "appId": "figstep",
                "name": "FigStep: Visual Prompt Injection",
                "description": "Embed adversarial instructions into images to bypass VLLM safety filters. Generate typographic attack images and test defenses.",
                "topics": ["Security", "Vision LLM", "Prompt Injection", "FigStep"],
                "duration": "20-30 Minutes",
            },
            {
                "order": "02",
                "id": "imscaler",
                "appId": "imscaler",
                "name": "IMScaler: Anamorpher Image Attack",
                "description": "Exploit VLLM image preprocessing pipelines using scale-dependent adversarial images. Generate anamorpher attacks and test defenses against resolution-dependent exploits.",
                "topics": ["Security", "Vision LLM", "Anamorpher", "Image Scaling"],
                "duration": "20-30 Minutes",
            },
            {
                "order": "03",
                "id": "cursedpixels",
                "appId": "cursedpixels",
                "name": "Cursed Pixels: PGD Pixel Manipulation",
                "description": "Run a Projected Gradient Descent attack on a vision-language model. Watch losses, perturbation, and live model output evolve as imperceptible pixel changes flip the response.",
                "topics": ["Security", "Vision LLM", "Adversarial", "PGD"],
                "duration": "20-30 Minutes",
            },
        ],
        "ai-supply-chain-attacks": [  # AI Supply Chain Attacks
            {
                "order": "01",
                "id": "backdoorcheckpoint",
                "appId": "backdoorcheckpoint",
                "name": "Backdoored Checkpoint: LoRA Trojan",
                "description": "Implant a visual-trigger backdoor into SmolVLM via LoRA fine-tuning. Watch clean and poisoned losses diverge as the watermark trigger learns to redirect outputs to a fixed payload.",
                "topics": ["Security", "Supply Chain", "Backdoor", "LoRA", "Vision LLM"],
                "duration": "20-30 Minutes",
            },
            {
                "order": "02",
                "id": "poisoneddataset",
                "appId": "poisoneddataset",
                "name": "Poisoned Dataset: Trojaning From Scratch",
                "description": "Train a tiny vision-LLM (ViT + BERT decoder) from scratch on a partly-poisoned dataset. Watch the backdoor implant during training and run a caption-frequency anomaly defense.",
                "topics": ["Security", "Supply Chain", "Backdoor", "Data Poisoning", "Training"],
                "duration": "25-40 Minutes",
            },
        ],
    },
}
