"""
메시지 버스
에이전트 간 통신을 위한 메시지 큐 시스템
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict

from ..utils.logging_utils import setup_logger, get_logger


class MessageType(Enum):
    """메시지 유형"""
    # 지식 관련
    KNOWLEDGE_REQUEST = "knowledge_request"
    KNOWLEDGE_RESPONSE = "knowledge_response"
    
    # 컨텍스트 관련 
    CONTEXT_REQUEST = "context_request"
    CONTEXT_RESPONSE = "context_response"
    HISTORY_REQUEST = "history_request"
    HISTORY_RESPONSE = "history_response"
    
    # 추론 관련 
    REASONING_REQUEST = "reasoning_request"
    REASONING_RESPONSE = "reasoning_response"
    
    # 학습 관련 
    LEARNING_REQUEST = "learning_request"
    LEARNING_RESPONSE = "learning_response"
    TASK_MODULE_REQUEST = "task_module_request"
    
    # 시스템 관련 
    TASK_ASSIGNMENT = "task_assignment"
    TASK_COMPLETE = "task_complete"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class Message:
    """에이전트 간 통신 메시지"""
    
    type: MessageType
    sender: str
    receiver: str
    payload: Dict[str, Any]
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None  # 요청-응답 매핑용
    priority: int = 1  # 1(낮음) ~ 5(높음)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "message_id": self.message_id,
            "type": self.type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "priority": self.priority,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """딕셔너리에서 생성"""
        return cls(
            message_id=data["message_id"],
            type=MessageType(data["type"]),
            sender=data["sender"],
            receiver=data["receiver"],
            payload=data["payload"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            correlation_id=data.get("correlation_id"),
            priority=data.get("priority", 1),
        )
    
    def create_response(
        self, 
        response_type: MessageType, 
        payload: Dict[str, Any]
    ) -> "Message":
        """응답 메시지 생성"""
        return Message(
            type=response_type,
            sender=self.receiver,
            receiver=self.sender,
            payload=payload,
            correlation_id=self.message_id,
        )


class MessageBus:
    """
    에이전트 간 메시지 버스
    
    비동기 pub-sub 패턴을 사용하여 에이전트 간 통신 지원
    """
    
    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        self.queues: Dict[str, asyncio.Queue] = {}
        self.subscribers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.pending_responses: Dict[str, asyncio.Future] = {}
        self.logger = setup_logger("agi_framework.message_bus")
        self._running = False
    
    def register_agent(self, agent_id: str) -> asyncio.Queue:
        """
        에이전트 등록
        
        Args:
            agent_id: 에이전트 ID
            
        Returns:
            에이전트 전용 메시지 큐
        """
        if agent_id not in self.queues:
            self.queues[agent_id] = asyncio.Queue(maxsize=self.max_queue_size)
            self.logger.info(f"에이전트 등록: {agent_id}")
        return self.queues[agent_id]
    
    def unregister_agent(self, agent_id: str):
        """에이전트 등록 해제"""
        if agent_id in self.queues:
            del self.queues[agent_id]
            self.logger.info(f"에이전트 등록 해제: {agent_id}")
    
    def subscribe(self, message_type: MessageType, callback: Callable):
        """
        메시지 유형 구독
        
        Args:
            message_type: 구독할 메시지 유형
            callback: 메시지 수신 시 호출할 콜백
        """
        self.subscribers[message_type].append(callback)
        self.logger.debug(f"구독 등록: {message_type.value}")
    
    async def publish(self, message: Message):
        """
        메시지 발행
        
        Args:
            message: 발행할 메시지
        """
        # 특정 수신자에게 전송
        if message.receiver in self.queues:
            try:
                await self.queues[message.receiver].put(message)
                self.logger.debug(
                    f"메시지 전송: {message.type.value} "
                    f"({message.sender} -> {message.receiver})"
                )
            except asyncio.QueueFull:
                self.logger.error(f"큐 가득 참: {message.receiver}")
                raise
        else:
            self.logger.warning(f"수신자 없음: {message.receiver}")
        
        # 구독자들에게 알림
        for callback in self.subscribers.get(message.type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                self.logger.error(f"구독자 콜백 오류: {e}")
    
    async def request(
        self, 
        message: Message, 
        timeout: float = 30.0
    ) -> Optional[Message]:
        """
        요청-응답 패턴으로 메시지 전송
        
        Args:
            message: 요청 메시지
            timeout: 응답 대기 시간 (초)
            
        Returns:
            응답 메시지 또는 None (타임아웃)
        """
        # Future 생성
        future = asyncio.get_event_loop().create_future()
        self.pending_responses[message.message_id] = future
        
        # 메시지 발행
        await self.publish(message)
        
        try:
            # 응답 대기
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            self.logger.warning(f"응답 타임아웃: {message.message_id}")
            return None
        finally:
            # 정리
            self.pending_responses.pop(message.message_id, None)
    
    async def respond(self, response: Message):
        """
        응답 메시지 처리
        
        Args:
            response: 응답 메시지
        """
        if response.correlation_id in self.pending_responses:
            future = self.pending_responses[response.correlation_id]
            if not future.done():
                future.set_result(response)
        
        # 일반 발행도 수행
        await self.publish(response)
    
    async def broadcast(self, message: Message, exclude: List[str] = None):
        """
        모든 에이전트에게 브로드캐스트
        
        Args:
            message: 브로드캐스트할 메시지
            exclude: 제외할 에이전트 ID 목록
        """
        exclude = exclude or []
        for agent_id in self.queues:
            if agent_id not in exclude:
                broadcast_msg = Message(
                    type=message.type,
                    sender=message.sender,
                    receiver=agent_id,
                    payload=message.payload,
                    priority=message.priority,
                )
                await self.publish(broadcast_msg)
    
    def get_queue_status(self) -> Dict[str, int]:
        """각 에이전트 큐 상태 반환"""
        return {
            agent_id: queue.qsize() 
            for agent_id, queue in self.queues.items()
        }
    
    async def drain_queue(self, agent_id: str) -> List[Message]:
        """에이전트 큐 비우기"""
        messages = []
        if agent_id in self.queues:
            queue = self.queues[agent_id]
            while not queue.empty():
                try:
                    msg = queue.get_nowait()
                    messages.append(msg)
                except asyncio.QueueEmpty:
                    break
        return messages

