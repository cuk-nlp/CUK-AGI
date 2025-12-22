"""
학습 에이전트
자기 평가 기반 강화학습 루프 및 태스크 모듈 학습
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .base_agent import BaseAgent, AgentConfig, TaskResult
from ..core.message_bus import Message, MessageBus, MessageType
from ..models.reward_model import RewardModel, PreferenceDataset, PPOTrainer


@dataclass
class EvaluationResult:
    """평가 결과"""
    prompt: str
    response: str
    score: float
    feedback: str
    aspects: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LearningConfig(AgentConfig):
    """학습 에이전트 설정"""
    reward_model_path: str = None
    preference_data_path: str = None
    evaluation_frequency: int = 100
    learning_rate: float = 1e-5
    batch_size: int = 32
    ppo_epochs: int = 4
    auto_learn: bool = True  # 자동 학습 활성화


class LearningAgent(BaseAgent):
    """
    학습 에이전트
    
    주요 기능:
    1. 자기 평가 기반 피드백 생성
    2. 선호도 데이터 수집 및 관리
    3. 리워드 모델 학습
    4. PPO 기반 강화학습 루프
    5. 태스크 모듈 업데이트 요청
    """
    
    def __init__(
        self,
        message_bus: MessageBus,
        config: LearningConfig = None,
        agent_id: str = None
    ):
        config = config or LearningConfig(
            name="learning",
            description="자기 평가 기반 강화학습 에이전트"
        )
        super().__init__(config, message_bus, agent_id)
        
        self.learn_config = config
        
        # 리워드 모델
        self.reward_model = RewardModel(learning_rate=config.learning_rate)
        
        # 선호도 데이터셋
        self.preference_dataset = PreferenceDataset(config.preference_data_path)
        
        # PPO 트레이너
        self.ppo_trainer = PPOTrainer(
            reward_model=self.reward_model,
            n_epochs=config.ppo_epochs,
        )
        
        # 평가 이력
        self.evaluation_history: List[EvaluationResult] = []
        
        # 학습 통계
        self.learning_stats = {
            "total_evaluations": 0,
            "total_preferences": 0,
            "reward_model_updates": 0,
            "policy_updates": 0,
        }
        
        # 핸들러 등록
        self.register_handler(
            MessageType.LEARNING_REQUEST,
            self._handle_learning_request
        )
    
    async def on_start(self):
        """시작 시 초기화"""
        self.logger.info("학습 에이전트 초기화 중...")
        
        try:
            self.reward_model.initialize()
            self.logger.info("리워드 모델 초기화 완료")
        except Exception as e:
            self.logger.warning(f"리워드 모델 초기화 실패: {e}")
        
        # 자동 학습 루프 시작
        if self.learn_config.auto_learn:
            asyncio.create_task(self._auto_learning_loop())
    
    async def _auto_learning_loop(self):
        """주기적으로 데이터셋 크기를 확인하여 임계치 초과 시 자동 학습 수행"""
        while self._running:
            try:
                # 주기적 체크
                await asyncio.sleep(60)  # 1분마다
                
                # 충분한 데이터가 쌓이면 학습
                if len(self.preference_dataset) >= self.learn_config.batch_size:
                    await self._trigger_learning()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"자동 학습 오류: {e}")
    
    async def _trigger_learning(self):
        """학습 트리거"""
        self.logger.info("자동 학습 시작...")
        
        # 리워드 모델 학습
        metrics = self.reward_model.train_step(
            self.preference_dataset.get_batch(self.learn_config.batch_size)
        )
        self.learning_stats["reward_model_updates"] += 1
        
        # PPO 업데이트
        if self.ppo_trainer.experience_buffer:
            ppo_metrics = self.ppo_trainer.update_policy()
            self.learning_stats["policy_updates"] += 1
            self.logger.info(f"PPO 업데이트: {ppo_metrics}")
        
        self.logger.info(f"학습 완료: {metrics}")
    
    async def handle_message(self, message: Message) -> TaskResult:
        """기본 메시지 처리"""
        return TaskResult(
            success=False,
            error=f"처리되지 않은 메시지 유형: {message.type.value}"
        )
    
    async def _handle_learning_request(self, message: Message) -> TaskResult:
        """
        학습 요청 처리
        
        Payload:
            action: evaluate, add_preference, train, get_stats
            data: 액션별 데이터
        """
        action = message.payload.get("action", "evaluate")
        data = message.payload.get("data", {})
        
        self.logger.log_action("LEARNING", {"action": action})
        
        try:
            if action == "evaluate":
                # 자기 평가 수행
                result = await self.evaluate_response(
                    prompt=data.get("prompt", ""),
                    response=data.get("response", ""),
                    context=data.get("context", {}),
                )
                return TaskResult(
                    success=True,
                    result={
                        "score": result.score,
                        "feedback": result.feedback,
                        "aspects": result.aspects,
                    }
                )
            
            elif action == "add_preference":
                # 선호도 데이터 추가
                self.add_preference(
                    prompt=data.get("prompt", ""),
                    chosen=data.get("chosen", ""),
                    rejected=data.get("rejected", ""),
                    metadata=data.get("metadata", {}),
                )
                return TaskResult(
                    success=True,
                    result={"total_preferences": len(self.preference_dataset)}
                )
            
            elif action == "train":
                # 수동 학습 트리거
                epochs = data.get("epochs", 10)
                metrics = self.train_reward_model(epochs=epochs)
                return TaskResult(
                    success=True,
                    result={"training_metrics": metrics}
                )
            
            elif action == "get_stats":
                # 통계 반환
                return TaskResult(
                    success=True,
                    result=self.get_learning_statistics()
                )
            
            elif action == "add_experience":
                # RL 경험 추가
                self.ppo_trainer.add_experience(
                    prompt=data.get("prompt", ""),
                    response=data.get("response", ""),
                    reward=data.get("reward"),
                )
                return TaskResult(
                    success=True,
                    result=self.ppo_trainer.get_statistics()
                )
            
            else:
                return TaskResult(
                    success=False,
                    error=f"알 수 없는 액션: {action}"
                )
                
        except Exception as e:
            self.logger.error(f"학습 요청 처리 오류: {e}")
            return TaskResult(success=False, error=str(e))
    
    async def evaluate_response(
        self,
        prompt: str,
        response: str,
        context: Dict[str, Any] = None
    ) -> EvaluationResult:
        """
        응답 자기 평가
        
        <think>
        자기 평가는 여러 측면에서 응답의 품질을 평가합니다:
        1. 관련성: 프롬프트와의 관련도
        2. 정확성: 정보의 정확성 (검증 가능한 경우)
        3. 완전성: 답변의 완전성
        4. 명확성: 표현의 명확성
        5. 유용성: 사용자에게 실질적으로 도움이 되는지
        
        실제 시스템에서는 별도의 평가 LLM을 사용하거나
        다양한 휴리스틱을 조합하여 평가합니다.
        </think>
        
        Args:
            prompt: 원본 프롬프트
            response: 평가할 응답
            context: 추가 컨텍스트
            
        Returns:
            평가 결과
        """
        # 리워드 모델로 기본 점수 계산
        base_score = self.reward_model.compute_reward(prompt, response)
        
        # 세부 측면 평가
        aspects = self._evaluate_aspects(prompt, response, context or {})
        
        # 종합 점수 (가중 평균)
        weights = {
            "relevance": 0.3,
            "completeness": 0.25,
            "clarity": 0.2,
            "helpfulness": 0.25,
        }
        
        weighted_score = sum(
            aspects.get(k, 0.5) * w 
            for k, w in weights.items()
        )
        
        # 기본 점수와 세부 평가 결합
        final_score = 0.6 * weighted_score + 0.4 * ((base_score + 1) / 2)  # 0~1 범위로
        
        # 피드백 생성
        feedback = self._generate_feedback(aspects, final_score)
        
        result = EvaluationResult(
            prompt=prompt,
            response=response,
            score=final_score,
            feedback=feedback,
            aspects=aspects,
        )
        
        # 이력 저장
        self.evaluation_history.append(result)
        self.learning_stats["total_evaluations"] += 1
        
        # RL 경험 추가
        self.ppo_trainer.add_experience(
            prompt=prompt,
            response=response,
            reward=final_score * 2 - 1,  # -1 ~ 1 범위로 변환
        )
        
        return result
    
    def _evaluate_aspects(
        self,
        prompt: str,
        response: str,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """세부 측면 평가"""
        aspects = {}
        
        # 관련성 평가 (단어 겹침 기반 - 실제로는 더 정교한 방법 사용)
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        overlap = len(prompt_words & response_words) / max(len(prompt_words), 1)
        aspects["relevance"] = min(0.5 + overlap, 1.0)
        
        # 완전성 평가 (길이 기반 휴리스틱)
        if len(response) < 50:
            aspects["completeness"] = 0.3
        elif len(response) < 200:
            aspects["completeness"] = 0.6
        elif len(response) < 1000:
            aspects["completeness"] = 0.8
        else:
            aspects["completeness"] = 0.7  # 너무 길면 감점
        
        # 명확성 평가 (문장 구조 기반)
        sentences = response.split(".")
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        if 10 < avg_sentence_length < 25:
            aspects["clarity"] = 0.8
        elif avg_sentence_length < 5 or avg_sentence_length > 40:
            aspects["clarity"] = 0.4
        else:
            aspects["clarity"] = 0.6
        
        # 유용성 평가 (키워드 기반)
        helpful_indicators = ["따라서", "결론적으로", "예를 들어", "구체적으로", "방법"]
        unhelpful_indicators = ["모르", "불가능", "없습니다", "아닙니다"]
        
        helpful_count = sum(1 for ind in helpful_indicators if ind in response)
        unhelpful_count = sum(1 for ind in unhelpful_indicators if ind in response)
        
        aspects["helpfulness"] = 0.5 + 0.1 * helpful_count - 0.15 * unhelpful_count
        aspects["helpfulness"] = max(0, min(1, aspects["helpfulness"]))
        
        return aspects
    
    def _generate_feedback(
        self,
        aspects: Dict[str, float],
        score: float
    ) -> str:
        """평가 피드백 생성"""
        feedback_parts = []
        
        if score >= 0.8:
            feedback_parts.append("전반적으로 우수한 응답입니다.")
        elif score >= 0.6:
            feedback_parts.append("양호한 응답이지만 개선의 여지가 있습니다.")
        elif score >= 0.4:
            feedback_parts.append("보통 수준의 응답입니다.")
        else:
            feedback_parts.append("응답 품질 개선이 필요합니다.")
        
        # 세부 피드백
        for aspect, value in aspects.items():
            if value < 0.5:
                if aspect == "relevance":
                    feedback_parts.append("- 질문과의 관련성을 높여주세요.")
                elif aspect == "completeness":
                    feedback_parts.append("- 더 상세한 설명이 필요합니다.")
                elif aspect == "clarity":
                    feedback_parts.append("- 표현을 더 명확히 해주세요.")
                elif aspect == "helpfulness":
                    feedback_parts.append("- 실용적인 정보를 추가해주세요.")
        
        return " ".join(feedback_parts)
    
    # 외부 API
    
    def add_preference(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        metadata: Dict[str, Any] = None
    ):
        """선호도 데이터 추가"""
        self.preference_dataset.add(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            metadata=metadata,
        )
        self.learning_stats["total_preferences"] += 1
        self.logger.debug(f"선호도 데이터 추가 (총 {len(self.preference_dataset)}개)")
    
    def train_reward_model(
        self,
        epochs: int = 10,
        batch_size: int = None
    ) -> List[Dict[str, float]]:
        """리워드 모델 수동 학습"""
        batch_size = batch_size or self.learn_config.batch_size
        
        metrics = self.reward_model.train(
            dataset=self.preference_dataset,
            epochs=epochs,
            batch_size=batch_size,
        )
        
        self.learning_stats["reward_model_updates"] += epochs
        
        return metrics
    
    def update_policy(self) -> Dict[str, float]:
        """정책 업데이트"""
        metrics = self.ppo_trainer.update_policy()
        self.learning_stats["policy_updates"] += 1
        return metrics
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """학습 통계 반환"""
        return {
            "general": self.learning_stats,
            "preference_dataset": self.preference_dataset.statistics(),
            "ppo_buffer": self.ppo_trainer.get_statistics(),
            "recent_evaluations": [
                {
                    "score": e.score,
                    "aspects": e.aspects,
                    "timestamp": e.timestamp.isoformat(),
                }
                for e in self.evaluation_history[-10:]  # 최근 10개
            ],
        }
    
    async def request_task_module_update(
        self,
        module_name: str,
        training_data: List[Dict[str, Any]]
    ):
        """
        추론 에이전트에 태스크 모듈 업데이트 요청
        
        Args:
            module_name: 업데이트할 모듈 이름
            training_data: 학습 데이터
        """
        reasoning_agent_id = "reasoning"  # 실제로는 레지스트리에서 조회
        
        await self.send_message(
            receiver=reasoning_agent_id,
            message_type=MessageType.TASK_MODULE_REQUEST,
            payload={
                "action": "update",
                "module_name": module_name,
                "training_data": training_data,
            },
        )
        
        self.logger.info(f"태스크 모듈 업데이트 요청: {module_name}")
    
    def save_state(self, path: str):
        """상태 저장"""
        self.reward_model.save(f"{path}/reward_model.pt")
        self.preference_dataset.save(f"{path}/preferences.json")
        self.logger.info(f"학습 상태 저장: {path}")
    
    def load_state(self, path: str):
        """상태 로드"""
        try:
            self.reward_model.load(f"{path}/reward_model.pt")
            self.preference_dataset.load(f"{path}/preferences.json")
            self.logger.info(f"학습 상태 로드: {path}")
        except Exception as e:
            self.logger.warning(f"상태 로드 실패: {e}")

