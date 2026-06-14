from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from typing import Any, List, Optional
from utils.key_manager import key_manager

class RotatingGemini(BaseChatModel):
    """요청 시마다 다음 키를 사용하는 Gemini 2.5 Flash LLM"""
    model_name: str = "gemini-2.5-flash"  # Gemini 2.5 Flash 사용
    temperature: float = 0.1

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=key_manager.get_next_key(),
            temperature=self.temperature,
            convert_system_message_to_human=True
        )
        return llm._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=key_manager.get_next_key(),
            temperature=self.temperature,
            convert_system_message_to_human=True
        )
        return await llm._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "rotating-gemini"

def get_gemini():
    """기존 코드와의 호환을 위한 인스턴스 반환"""
    return RotatingGemini()
