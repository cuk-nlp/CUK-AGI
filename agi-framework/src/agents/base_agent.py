"""
기본 에이전트 클래스
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..core.message_bus import Message, MessageBus, MessageType
from ..utils.logging_utils import AgentLogger


class AgentState(Enum):
    """에이전트 상태"""
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class AgentConfig:
    """에이전트 설정"""
    name: str
    description: str = ""
    max_concurrent_tasks: int = 5
    timeout: float = 60.0
    retry_attempts: int = 3


@dataclass
class TaskResult:
    """태스크 실행 결과"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    기본 에이전트 클래스
    
    모든 에이전트는 이 클래스를 상속받아 구현
    비동기 메시지 처리 및 상태 관리 기능 제공
    """
    
    def __init__(
        self, 
        config: AgentConfig, 
        message_bus: MessageBus,
        agent_id: Optional[str] = None
    ):
        self.config = config
        self.message_bus = message_bus
        self.agent_id = agent_id or f"{config.name}_{uuid.uuid4().hex[:8]}"
        self.state = AgentState.IDLE
        self.logger = AgentLogger(config.name, self.agent_id)
        
        # 메시지 큐 등록
        self.message_queue = message_bus.register_agent(self.agent_id)
        
        # 내부 상태
        self._running = False
        self._task_count = 0
        self._created_at = datetime.now()
        self._last_activity = datetime.now()
        
        # 메시지 핸들러 등록
        self._message_handlers: Dict[MessageType, callable] = {}
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """기본 메시지 핸들러 등록"""
        self._message_handlers[MessageType.HEARTBEAT] = self._handle_heartbeat
        self._message_handlers[MessageType.ERROR] = self._handle_error
    
    def register_handler(self, message_type: MessageType, handler: callable):
        """메시지 핸들러 등록"""
        self._message_handlers[message_type] = handler
        self.logger.debug(f"핸들러 등록: {message_type.value}")
    
    async def _handle_heartbeat(self, message: Message) -> TaskResult:
        """하트비트 처리"""
        return TaskResult(
            success=True,
            result={
                "agent_id": self.agent_id,
                "state": self.state.value,
                "uptime": (datetime.now() - self._created_at).total_seconds(),
            }
        )
    
    async def _handle_error(self, message: Message) -> TaskResult:
        """에러 메시지 처리"""
        self.logger.error(f"에러 수신: {message.payload}")
        return TaskResult(success=True)
    
    async def start(self):
        """에이전트 시작"""
        self._running = True
        self.state = AgentState.IDLE
        self.logger.info("에이전트 시작")
        
        # 초기화 훅
        await self.on_start()
        
        # 메시지 처리 루프
        asyncio.create_task(self._message_loop())
    
    async def stop(self):
        """에이전트 중지"""
        self._running = False
        self.state = AgentState.STOPPED
        self.logger.info("에이전트 중지")
        
        # 정리 훅
        await self.on_stop()
        
        # 메시지 버스에서 등록 해제
        self.message_bus.unregister_agent(self.agent_id)
    
    async def _message_loop(self):
        """메시지 처리 루프"""
        while self._running:
            try:
                # 메시지 대기 (타임아웃 포함)
                try:
                    message = await asyncio.wait_for(
                        self.message_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # 메시지 처리
                await self._process_message(message)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"메시지 루프 오류: {e}")
                self.state = AgentState.ERROR
    
    async def _process_message(self, message: Message):
        """메시지 처리"""
        self._last_activity = datetime.now()
        self.logger.log_message_received(message.type.value, message.sender)
        
        # 상태 업데이트
        prev_state = self.state
        self.state = AgentState.PROCESSING
        self._task_count += 1
        
        try:
            # 핸들러 찾기
            handler = self._message_handlers.get(message.type)
            
            if handler:
                result = await handler(message)
            else:
                # 기본 처리
                result = await self.handle_message(message)
            
            # 응답 필요시 전송
            if message.correlation_id is None and result.success:
                # 응답 메시지 생성 및 전송
                response_type = self._get_response_type(message.type)
                if response_type:
                    response = message.create_response(
                        response_type,
                        {"result": result.result, "metadata": result.metadata}
                    )
                    await self.message_bus.respond(response)
            
        except Exception as e:
            self.logger.error(f"메시지 처리 오류: {e}")
            self.state = AgentState.ERROR
            
            # 에러 응답
            error_response = message.create_response(
                MessageType.ERROR,
                {"error": str(e), "original_type": message.type.value}
            )
            await self.message_bus.respond(error_response)
        
        finally:
            self.state = prev_state if prev_state != AgentState.PROCESSING else AgentState.IDLE
    
    def _get_response_type(self, request_type: MessageType) -> Optional[MessageType]:
        """요청 유형에 대응하는 응답 유형 반환"""
        response_map = {
            MessageType.KNOWLEDGE_REQUEST: MessageType.KNOWLEDGE_RESPONSE,
            MessageType.CONTEXT_REQUEST: MessageType.CONTEXT_RESPONSE,
            MessageType.HISTORY_REQUEST: MessageType.HISTORY_RESPONSE,
            MessageType.REASONING_REQUEST: MessageType.REASONING_RESPONSE,
            MessageType.LEARNING_REQUEST: MessageType.LEARNING_RESPONSE,
        }
        return response_map.get(request_type)
    
    async def send_message(
        self, 
        receiver: str, 
        message_type: MessageType, 
        payload: Dict[str, Any],
        priority: int = 1
    ) -> Optional[Message]:
        """
        메시지 전송
        
        Args:
            receiver: 수신자 에이전트 ID
            message_type: 메시지 유형
            payload: 메시지 내용
            priority: 우선순위
            
        Returns:
            응답 메시지 (요청-응답 패턴의 경우)
        """
        message = Message(
            type=message_type,
            sender=self.agent_id,
            receiver=receiver,
            payload=payload,
            priority=priority,
        )
        
        self.logger.log_message_sent(message_type.value, receiver)
        
        # 응답이 필요한 메시지인지 확인
        if self._get_response_type(message_type):
            return await self.message_bus.request(message, timeout=self.config.timeout)
        else:
            await self.message_bus.publish(message)
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """에이전트 상태 반환"""
        return {
            "agent_id": self.agent_id,
            "name": self.config.name,
            "state": self.state.value,
            "task_count": self._task_count,
            "created_at": self._created_at.isoformat(),
            "last_activity": self._last_activity.isoformat(),
            "queue_size": self.message_queue.qsize(),
        }
    
    # 추상 메서드 - 서브클래스에서 구현
    
    @abstractmethod
    async def handle_message(self, message: Message) -> TaskResult:
        """
        메시지 처리 (서브클래스에서 구현)
        
        Args:
            message: 처리할 메시지
            
        Returns:
            처리 결과
        """
        pass
    
    async def on_start(self):
        """시작 시 호출되는 훅 (선택적 오버라이드)"""
        pass
    
    async def on_stop(self):
        """중지 시 호출되는 훅 (선택적 오버라이드)"""
        pass

