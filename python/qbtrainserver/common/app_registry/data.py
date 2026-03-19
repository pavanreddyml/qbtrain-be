"""
Standardized backend registry data
- order: numeric string used for sorting
- id: lowercase, no spaces/specials (used in URLs & images)
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
            "duration": "15-20 minutes",
            "topics": ["Systems", "LLMs"],
        },
        {
            "order": "02",
            "id": "foundations-2",
            "name": "Foundations 2",
            "description": "Advanced parameter interactions and complex sampling strategies.",
            "duration": "20-30 minutes",
            "topics": ["Hyperparameters", "Interactions"],
        },
    ],
    "ai-security": [
        {
            "order": "01",
            "id": "ai-security-basics",
            "name": "AI Security Basics",
            "description": "Hands-on labs for AI security basics.",
            "duration": "15-20 minutes",
            "topics": ["AI Security"],
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
                "topics": ["temperature", "sampling"],
                "duration": "15-20 minutes",
            },
            {
                "order": "02",
                "id": "topk-101",
                "appId": "topk-101",
                "name": "Top-K 101: Taming the Tail",
                "description": "Understand how Top-K trims unlikely tokens to prevent incoherence; greedy decoding vs. wide sampling.",
                "topics": ["top-k", "decoding"],
                "duration": "15-20 minutes",
            },
            {
                "order": "03",
                "id": "topp-101",
                "appId": "topp-101",
                "name": "Top-P 101: Nucleus Sampling",
                "description": "Learn Top-P; compare with Top-K & Temperature; when to use for natural-sounding text.",
                "topics": ["top-p", "nucleus"],
                "duration": "15-20 minutes",
            },
            {
                "order": "04",
                "id": "presence-penalty-101",
                "appId": "presence-penalty-101",
                "name": "Presence Penalty: Preventing Repetition",
                "description": "How presence penalty discourages repeating topics; control breadth of conversation.",
                "topics": ["penalties", "repetition"],
                "duration": "15-20 minutes",
            },
            {
                "order": "05",
                "id": "frequency-penalty-101",
                "appId": "frequency-penalty-101",
                "name": "Frequency Penalty: Word Repetition",
                "description": "Discourage repeating words; avoid stuttering or loops.",
                "topics": ["penalties", "frequency"],
                "duration": "15-20 minutes",
            },
        ],
        "foundations-2": [  # Foundations 2
            {
                "order": "01",
                "id": "temp-vs-topk",
                "appId": "temp-vs-topk",
                "name": "Temp vs Top-K: The Power Struggle",
                "description": "See interactions/overrides; why Top-K=1 makes Temperature irrelevant.",
                "topics": ["temperature", "top-k"],
                "duration": "20-30 minutes",
            },
            {
                "order": "02",
                "id": "temp-vs-topp",
                "appId": "temp-vs-topp",
                "name": "Temp vs Top-P: The Safety Net",
                "description": "Low Top-P can tame high Temperature; hard cut vs. cumulative sum trade-offs.",
                "topics": ["temperature", "top-p"],
                "duration": "20-30 minutes",
            },
            {
                "order": "03",
                "id": "topk-vs-topp",
                "appId": "topk-vs-topp",
                "name": "Top-K vs Top-P: The Filter Funnel",
                "description": "Double filter strategy; K acts coarse, P acts fine; order of operations.",
                "topics": ["top-k", "top-p"],
                "duration": "20-30 minutes",
            },
            {
                "order": "04",
                "id": "creativity-cocktail",
                "appId": "creativity-cocktail",
                "name": "The Creativity Cocktail",
                "description": "Explore Temperature + Frequency + Presence penalties; break loops without losing coherence.",
                "topics": ["temperature", "penalties"],
                "duration": "20-30 minutes",
            },
        ],
    },
    "ai-security": {  # AI Security
        "ai-security-basics": [  # AI Security Basics
            {
                "order": "01",
                "id": "crdlr",
                "appId": "crdlr",
                "name": "CRDLR: Adversarial Prompting Fundamentals",
                "description": "Fundamentals of adversarial prompting; attack vectors & mitigations; hands-on practice.",
                "topics": ["security", "prompting"],
                "duration": "15-25 minutes",
            },
            {
                "order": "02",
                "id": "echoleak",
                "appId": "echoleak",
                "name": "Echoleak: Information Disclosure",
                "description": "Master information extraction from multiple sources; understand data leakage vectors.",
                "topics": ["security", "information-disclosure"],
                "duration": "15-25 minutes",
            },
            # {
            #     "order": "03",
            #     "id": "codeexec",
            #     "appId": "codeexec",
            #     "name": "Code Executor: Query-Driven Code Generation",
            #     "description": "Execute queries by intelligently choosing between direct answers, web search, file analysis, and Python code execution.",
            #     "topics": ["security", "code-execution", "reasoning"],
            #     "duration": "20-30 minutes",
            # },
            # {
            #     "order": "04",
            #     "id": "modeltheft",
            #     "appId": "modeltheft",
            #     "name": "Model Theft: Knowledge Distillation Attack",
            #     "description": "Steal a teacher model's capabilities by distilling knowledge into a small student model. Explore distillation methods and observe the student improve.",
            #     "topics": ["security", "model-theft", "distillation"],
            #     "duration": "25-35 minutes",
            # },
        ],
    },
}