ENGLISH_REACT_COMPLETION_PROMPT_TEMPLATES = """Respond to the human as helpfully and accurately as possible. 

{{instruction}}

You have access to the following tools:

{{tools}}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
Valid "action" values: "Final Answer" or {{tool_names}}

Provide only ONE action per $JSON_BLOB, as shown:

```
{
  "action": $TOOL_NAME,
  "action_input": $ACTION_INPUT
}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{
  "action": "Final Answer",
  "action_input": "Final response to human"
}
```

Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.
{{historic_messages}}
Question: {{query}}
{{agent_scratchpad}}
Thought:"""

ENGLISH_REACT_COMPLETION_AGENT_SCRATCHPAD_TEMPLATES = """Observation: {{observation}}
Thought:"""

ENGLISH_REACT_CHAT_PROMPT_TEMPLATES = """Respond to the human as helpfully and accurately as possible. 

{{instruction}}

You have access to the following tools:

{{tools}}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
Valid "action" values: "Final Answer" or {{tool_names}}

Provide only ONE action per $JSON_BLOB, as shown:

```
{
  "action": $TOOL_NAME,
  "action_input": $ACTION_INPUT
}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{
  "action": "Final Answer",
  "action_input": "Final response to human"
}
```

Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.
"""

CH_REACT_CHAT_PROMPT_TEMPLATES_NYZ = """你善于理解用户的请求，并能够按照用户的要求，去主动思考和操作工具。记住，请尽量回复有用并且准确的信息！
以下是用户的提示词：
{{instruction}}

以下是你可以使用的工具:
{{tools}}

以下是你的操作规范：
你必须使用json blob来指定和描述一个“action”，json blob中需要包含 “action” 和 “action_input”信息，其中“action”可以是“Final Answer”或着｛｛tool_names｝｝中的一个。
每一个$JSON_BLOB只能用来描述一个action。JSON_BLOB示例如下:

```
{
  "action": $TOOL_NAME,
  "action_input": $ACTION_INPUT
}
```

以下是你的思考和操作步骤:

Question: 用户请求
Thought: 根据你已经知道的上下文信息，生成合适的action
Action: 
```
$JSON_BLOB
```
Observation: 收集和分析Action的执行结果
... (repeat Thought/Action/Observation N times)
Thought: 基于上下文信息，判断足以完成用户的需求，生成 “Final Answer” action
Action:
```
{
  "action": "Final Answer",
  "action_input": "Final response to human"
}
```

请注意：
1. 你生成的内容，必须始终是一个有效的json blob去指定一个action，否则你将受到惩罚；
2. 生成的“Final Answer”必须是一个action：```$JSON_BLOB```，否则你将受到惩罚；
3. 请用中文回应，否则你将受到惩罚；
4. 只能使用用户指定的tools，不能编造不存在的tool，否则你将受到惩罚；
5. 如有必要，使用工具；
6. 请严格遵守用户给定的提示词；
7. “Thought”的内容，只能是action的json blob；
8. 先执行action：```$JSON_BLOB```再执行observation：。
"""

ENGLISH_REACT_CHAT_AGENT_SCRATCHPAD_TEMPLATES = ""

REACT_PROMPT_TEMPLATES = {
    'english': {
        'chat': {
            'prompt': CH_REACT_CHAT_PROMPT_TEMPLATES_NYZ,
            'agent_scratchpad': ENGLISH_REACT_CHAT_AGENT_SCRATCHPAD_TEMPLATES
        },
        'completion': {
            'prompt': ENGLISH_REACT_COMPLETION_PROMPT_TEMPLATES,
            'agent_scratchpad': ENGLISH_REACT_COMPLETION_AGENT_SCRATCHPAD_TEMPLATES
        }
    }
}