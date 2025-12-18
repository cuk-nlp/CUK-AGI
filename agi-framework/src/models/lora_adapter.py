"""
태스크별 경량 파인튜닝을 위한 LoRA 어댑터 관리
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

# PyTorch 선택적 import
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# PEFT 선택적 import
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        PeftModel,
        TaskType,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


@dataclass
class AdapterConfig:
    """LoRA 어댑터 설정"""
    adapter_name: str
    task_type: str = "CAUSAL_LM"  # CAUSAL_LM, SEQ_2_SEQ_LM, etc.
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj"
    ])
    bias: str = "none"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "task_type": self.task_type,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdapterConfig":
        return cls(**data)
    
    def to_peft_config(self) -> "LoraConfig":
        """PEFT LoraConfig로 변환"""
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT가 필요합니다: pip install peft")
        
        return LoraConfig(
            task_type=TaskType[self.task_type],
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            bias=self.bias,
        )

class LoRAAdapter:
    """
    LoRA 어댑터 관리자
    
    태스크별 어댑터 로드/저장 및 동적 전환 지원
    """
    
    def __init__(
        self,
        base_model_name: str = "meta-llama/Llama-2-7b-hf",
        adapter_dir: str = "./adapters",
        device: str = "auto"
    ):
        self.base_model_name = base_model_name
        self.adapter_dir = Path(adapter_dir)
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        self.base_model = None
        self.tokenizer = None
        self.loaded_adapters: Dict[str, AdapterConfig] = {}
        self.active_adapter: Optional[str] = None
        
        self._initialized = False
    
    def initialize(self, load_base_model: bool = True):
        """
        초기화
        
        Args:
            load_base_model: 기본 모델 로드 여부 (False면 시뮬레이션 모드)
        """
        if self._initialized:
            return
        
        if load_base_model and TORCH_AVAILABLE:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model_name,
                    trust_remote_code=True,
                )
                
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map=self.device,
                    trust_remote_code=True,
                )
                
                print(f"기본 모델 로드 완료: {self.base_model_name}")
                
            except Exception as e:
                print(f"모델 로드 실패 (시뮬레이션 모드): {e}")
        
        self._initialized = True
    
    def create_adapter(
        self,
        adapter_name: str,
        config: AdapterConfig = None
    ) -> AdapterConfig:
        """
        새 어댑터 생성
        
        Args:
            adapter_name: 어댑터 이름
            config: 어댑터 설정
            
        Returns:
            생성된 어댑터 설정
        """
        if config is None:
            config = AdapterConfig(adapter_name=adapter_name)
        
        if self.base_model is not None and PEFT_AVAILABLE:
            peft_config = config.to_peft_config()
            self.base_model = get_peft_model(self.base_model, peft_config)
        
        self.loaded_adapters[adapter_name] = config
        self.active_adapter = adapter_name
        
        return config
    
    def load_adapter(self, adapter_path: str) -> Optional[str]:
        """
    
        
        Args:
            adapter_path: 어댑터 경로
            
        Returns:
            로드된 어댑터 이름
        """
        adapter_path = Path(adapter_path)
        config_path = adapter_path / "adapter_config.json"
        
        if not config_path.exists():
            print(f"어댑터 설정 없음: {config_path}")
            return None
        
        with open(config_path) as f:
            config_data = json.load(f)
        
        config = AdapterConfig.from_dict(config_data)
        
        if self.base_model is not None and PEFT_AVAILABLE:
            self.base_model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
            )
        
        self.loaded_adapters[config.adapter_name] = config
        self.active_adapter = config.adapter_name
        
        return config.adapter_name
    
    def save_adapter(self, adapter_name: str, path: str = None):
        """
        어댑터 저장
        
        Args:
            adapter_name: 어댑터 이름
            path: 저장 경로
        """
        if adapter_name not in self.loaded_adapters:
            raise ValueError(f"어댑터 없음: {adapter_name}")
        
        save_path = Path(path) if path else self.adapter_dir / adapter_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 설정 저장
        config = self.loaded_adapters[adapter_name]
        with open(save_path / "adapter_config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        
        # 모델 가중치 저장
        if self.base_model is not None and PEFT_AVAILABLE:
            self.base_model.save_pretrained(save_path)
    
    def switch_adapter(self, adapter_name: str):
        """
        활성 어댑터 전환
        
        Args:
            adapter_name: 전환할 어댑터 이름
        """
        if adapter_name not in self.loaded_adapters:
            raise ValueError(f"어댑터 없음: {adapter_name}")
        
        if self.base_model is not None and PEFT_AVAILABLE:
            self.base_model.set_adapter(adapter_name)
        
        self.active_adapter = adapter_name
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
            max_length: 최대 길이
            temperature: 온도
            top_p: Top-p 샘플링
            
        Returns:
            생성된 텍스트
        """
        if self.base_model is None or self.tokenizer is None:
            # 시뮬레이션 모드
            return self._simulate_generation(prompt)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.base_model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 제거
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()
        
        return generated
    
    def _simulate_generation(self, prompt: str) -> str:
        """시뮬레이션 생성 (모델 없이 테스트용)"""
        adapter_info = f"[Adapter: {self.active_adapter or 'none'}]"
        
        if "교육자료" in prompt or "설명" in prompt:
            return f"{adapter_info} 이것은 시뮬레이션된 교육 자료 응답입니다. 실제 모델이 로드되면 더 상세한 설명이 생성됩니다."
        elif "소견" in prompt or "분석" in prompt:
            return f"{adapter_info} 시뮬레이션된 분석 소견입니다. 입력된 내용에 대한 전문적인 분석이 필요합니다."
        else:
            return f"{adapter_info} 쿼리: '{prompt[:50]}...'에 대한 시뮬레이션 응답입니다."
    
    def get_adapter_info(self) -> Dict[str, Any]:
        """어댑터 정보 반환"""
        return {
            "base_model": self.base_model_name,
            "loaded_adapters": list(self.loaded_adapters.keys()),
            "active_adapter": self.active_adapter,
            "model_loaded": self.base_model is not None,
        }
    
    def list_available_adapters(self) -> List[str]:
        """사용 가능한 어댑터 목록"""
        adapters = list(self.loaded_adapters.keys())
        
        # 저장된 어댑터 검색
        for path in self.adapter_dir.iterdir():
            if path.is_dir() and (path / "adapter_config.json").exists():
                if path.name not in adapters:
                    adapters.append(path.name)
        
        return adapters


class AdapterPool:
    """
    어댑터 풀
    
    여러 태스크별 어댑터를 관리하고 동적으로 전환
    """
    def __init__(self, base_model_name: str = None):
        self.base_model_name = base_model_name
        self.adapters: Dict[str, LoRAAdapter] = {}
        self.default_configs: Dict[str, AdapterConfig] = {
            "education_material": AdapterConfig(
                adapter_name="education_material",
                rank=16,
                alpha=32,
            ),
            "opinion_generator": AdapterConfig(
                adapter_name="opinion_generator",
                rank=8,
                alpha=16,
            ),
            "summarizer": AdapterConfig(
                adapter_name="summarizer",
                rank=8,
                alpha=16,
            ),
        }
    
    def get_adapter(self, task_name: str) -> LoRAAdapter:
        """태스크별 어댑터 획득"""
        if task_name not in self.adapters:
            adapter = LoRAAdapter(
                base_model_name=self.base_model_name,
                adapter_dir=f"./adapters/{task_name}",
            )
            
            config = self.default_configs.get(
                task_name,
                AdapterConfig(adapter_name=task_name)
            )
            
            adapter.create_adapter(task_name, config)
            self.adapters[task_name] = adapter
        
        return self.adapters[task_name]
    
    def list_tasks(self) -> List[str]:
        """등록된 태스크 목록"""
        return list(set(self.adapters.keys()) | set(self.default_configs.keys()))

