# python/qbtrainserver/common/app_registry/data.py

CATEGORIES = [
    {
        "id": "01",
        "name": "AI",
        "image": "ai_category.png",
        "description": "Hands-on labs for core AI/LLM product skills.",
    },
]

SECTION_METADATA = {
    "Foundations": {"duration": "15-20 minutes", "topics": ["Systems", "LLMs"]},
    "Foundations 2": {"duration": "20-30 minutes", "topics": ["Hyperparameters", "Interactions"]},
}

DEFAULT_APP_META = {"duration": "15-25 minutes", "topics": ["LLMs", "Applied AI"]}

APPS = {
    "AI": [
        {
            "id": "01.01",
            "subcategory": "Foundations",
            "description": "Understand tokens, context windows, embeddings, and sampling.",
            "apps": [
                {
                    "id": "01.01.01",
                    "appId": "temperature-101",
                    "name": "Temperature 101: From Logic to Chaos",
                    "image": "app_temperature_101.png",
                    "description": "Master the temperature hyperparameter through an interactive guided tour.\nObserve how randomness affects creativity, logic, and hallucination.\nLearn safe ranges for coding vs. creative writing tasks.\nDeliver a deep understanding of sampling strategies.",
                },
                {
                    "id": "01.01.02",
                    "appId": "topk-101",
                    "name": "Top-K 101: Taming the Tail",
                    "image": "app_topk_101.png",
                    "description": "Understand how Top-K cuts off unlikely tokens to prevent incoherence.\nExperiment with greedy decoding (K=1) vs wide sampling.\nLearn when to use Top-K instead of (or with) Temperature.\nControl the vocabulary pool size for your model.",
                },
                {
                    "id": "01.01.03",
                    "appId": "topp-101",
                    "name": "Top-P 101: Nucleus Sampling",
                    "image": "app_topp_101.png",
                    "description": "Learn how Top-P (Nucleus Sampling) dynamically adjusts the token pool.\nCompare it with Top-K and Temperature.\nUnderstand when to use it for more natural-sounding text.",
                },
                {
                    "id": "01.01.04",
                    "appId": "presence-penalty-101",
                    "name": "Presence Penalty: Preventing Repetition",
                    "image": "app_presence_101.png",
                    "description": "Explore how Presence Penalty discourages repeating topics.\nLearn to control the breadth of the conversation.",
                },
                {
                    "id":   "01.01.05",
                    "appId": "frequency-penalty-101",
                    "name": "Frequency Penalty: Word Repetition",
                    "image": "app_frequency_101.png",
                    "description": "Understand how Frequency Penalty discourages repeating specific words.\nFine-tune your model to avoid stuttering or loops.",
                },
            ],
        },
        {
            "id": "01.02",
            "subcategory": "Foundations 2",
            "description": "Advanced parameter interactions and complex sampling strategies.",
            "apps": [
                {
                    "id": "01.02.01",
                    "appId": "temp-vs-topk",
                    "name": "Temp vs Top-K: The Power Struggle",
                    "image": "app_temp_vs_topk.png",
                    "description": "Explore how Temperature and Top-K interact and override each other.\nSee why Top-K=1 makes Temperature irrelevant.\nLearn to balance creativity with stability using both parameters.\nDebug complex hallucination issues.",
                },
                {
                    "id": "01.02.02",
                    "appId": "temp-vs-topp",
                    "name": "Temp vs Top-P: The Safety Net",
                    "image": "app_temp_vs_topp.png",
                    "description": "Learn how Nucleus Sampling (Top-P) acts as a flexible safety net for Temperature.\nUnderstand why Low Top-P can tame High Temperature better than Top-K.\nExplore the trade-offs between 'Hard Cut' (Top-K) and 'Cumulative Sum' (Top-P).",
                },
                {
                    "id": "01.02.03",
                    "appId": "topk-vs-topp",
                    "name": "Top-K vs Top-P: The Filter Funnel",
                    "image": "app_topk_vs_topp.png",
                    "description": "Master the 'Double Filter' strategy.\nSee how Top-K acts as a coarse filter while Top-P acts as a fine filter.\nLearn the Order of Operations (K then P) and how to stack them for precision control.",
                },
                {
                    "id": "01.02.04",
                    "appId": "creativity-cocktail",
                    "name": "The Creativity Cocktail",
                    "image": "app_creativity_cocktail.png",
                    "description": "Experiment with the 3-way interaction of Temperature, Frequency Penalty, and Presence Penalty.\nSee how Penalties + High Temp force the model into 'The Void' (Hallucination).\nLearn to break loops without destroying coherence.",
                },
                
            ],
        },
    ]
}