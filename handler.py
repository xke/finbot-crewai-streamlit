import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler
from typing import Any, Dict
from agentops import record_function

#import weave

class CustomHandler(BaseCallbackHandler):
    """A custom handler for logging interactions within the process chain."""
    
    def __init__(self, agent_name: str) -> None:
        super().__init__()
        self.agent_name = agent_name

    #@weave.op()
    @record_function("on_chain_start")
    def on_chain_start(self, serialized: Dict[str, Any], outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Log the start of a chain with user input."""
        # comment out verbose chain input
        #st.session_state.messages.append({"role": "assistant", "content": outputs['input']})
        #st.chat_message("assistant").write(outputs['input'])
        
    #@weave.op()
    @record_function("on_agent_action")
    def on_agent_action(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """""Log the action taken by an agent during a chain run."""
        st.session_state.messages.append({"role": "assistant", "content": inputs['input']})
        st.chat_message("assistant").write(inputs['input'])
        
    #@weave.op()
    @record_function("on_chain_end")
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Log the end of a chain with the output generated by an agent."""
        message = f"##### {self.agent_name}:\n\n{outputs['output']}"
        st.session_state.messages.append({"role": self.agent_name, "content": message})
        st.chat_message(self.agent_name).write(message)
