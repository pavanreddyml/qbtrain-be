"""
Prompt templates for code execution app
"""

ROUTE_DECISION_PROMPT = """You are a query router that decides how to answer questions.

Based on the user's question and available context, choose ONE of these routes:
1. "direct" - Answer directly from knowledge (no code, search, or files needed)
2. "request" - Need to search the web for current information
3. "csv" - Need to analyze a CSV file provided by the user
4. "document" - Need to analyze a document (PDF, text, etc.) provided by the user
5. "python" - Need to write and execute Python code to compute/analyze/process data

User Question: {question}
Has CSV file: {has_csv}
Has document/PDF: {has_document}
Conversation history: {history_summary}

Respond with EXACTLY this JSON format (and nothing else):
{{
  "route": "<one of: direct, request, csv, document, python>",
  "reasoning": "<brief explanation why>"
}}"""

CODE_GEN_PROMPT = """You are a Python code generator for the {route} route.

User Question: {question}
Route: {route}
Context/File Content: {context}
Previous Output (if any): {previous_output}
Iteration: {iteration}

Generate Python code to answer this question. The code MUST:
1. Be syntactically correct Python
2. Print ONLY valid JSON to stdout in this format: {{"values": <answer>, "files": [<list of saved file paths>]}}
3. For {route} route:
   - "request": Use requests library to search/fetch web data
   - "csv": Parse and analyze the CSV data
   - "document": Parse and analyze the document
   - "python": Perform calculations, data processing, analysis
4. Handle errors gracefully
5. Save any generated graphs/images to files and list them in "files" array

IMPORTANT: The last line of code must print the JSON output. No other print statements.
Only respond with the Python code, no explanation or markdown."""

SUFFICIENCY_PROMPT = """Does this output sufficiently answer the user's question?

Question: {question}
Output: {output}
Iteration: {iteration}/3

Respond with EXACTLY this JSON:
{{"sufficient": <true or false>, "reason": "<explanation>"}}"""

DIRECT_ANSWER_PROMPT = """Answer this question directly based on your knowledge.

Question: {question}
Conversation history: {history}

Provide a clear, accurate, and concise answer."""

FINAL_ANSWER_PROMPT = """Synthesize this computational output into a clear answer to the user's question.

Question: {question}
Code Output: {output}
Route Used: {route}

Provide a well-formatted, readable answer that interprets and explains the output in context of the question."""

ROUTE_RETRY_PROMPT = """The previous {current_route} route did not produce sufficient output.

Question: {question}
Previous Output: {previous_output}
Previously Tried Routes: {tried_routes}

Suggest the BEST next route (one of: request, csv, document, python) that wasn't tried yet.
Respond with EXACTLY:
{{"next_route": "<route>", "reason": "<explanation>"}}"""
